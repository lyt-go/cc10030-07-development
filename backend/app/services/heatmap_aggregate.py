"""热力聚合领域服务：按网格层级和时间粒度对定位点进行聚合计算，生成热力聚合数据集。

在基础聚合能力之上，新增质检区间驱动的交付流程：

- 只从已核验（已确认）的质检区间准备输入并发起热力取数，未确认范围不会混入；
- 每次交付写入一份逐点可读回的输入凭据：内嵌点记录、逐点哈希与整体快照校验值；
- 原批次随后补点或修订时，已交付的统计可仅凭据重新计算，不依赖当前点文件内容。
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

from app.services.base import BaseModuleService
from app.store import store


class HeatmapAggregateService(BaseModuleService):
    MODULE = "heatmap_aggregate"
    ENTITY = "聚合窗口"
    OPERATION_TERM = "聚合"
    LIST_FIELDS = ["窗口编号", "网格层级", "时间粒度", "聚合指标", "落点总量", "峰值网格", "窗口类型", "聚合水位"]
    REQUIRED_FIELDS = ["窗口编号", "网格层级", "时间粒度"]
    STATUS_ORDER = ["草稿", "待聚合", "聚合中", "已产出"]
    ACTION_RULES = {"启动聚合": "聚合中", "确认产出": "已产出", "标记作废": "已产出"}
    NEGATIVE_ACTIONS = []

    # -- 已核验区间输入准备 --

    def prepare_verified_input(self, batch_key: str) -> dict[str, Any]:
        """从已核验区间筛选观测点，为热力取数准备输入文件。"""
        geo_svc = self._safe_import("app.services.geo_source", "GeoSourceService")()
        manifest = geo_svc.get_qc_manifest(batch_key)
        coverage = geo_svc.compute_coverage(batch_key)
        confirmed_ids = {
            pid for iv in manifest["intervals"] if iv["status"] == "已确认"
            for pid in iv["observed_point_ids"]
        }
        points, points_path = geo_svc.get_batch_points(batch_key)
        selected = [p for p in points if p.get("point_id") in confirmed_ids]
        for p in selected:
            p["qc_interval"] = self._interval_of(manifest, p.get("point_id"))

        output_file = self._data_dir("heatmap_input") / f"{batch_key}_verified.jsonl"
        # 每次准备都按当前已核验范围整份重写，避免追加模式残留未确认点
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for rec in selected:
                f.write(json_deterministic(rec) + "\n")
        written = len(selected)
        return {
            "batch_key": batch_key,
            "verified_point_count": written,
            "confirmed_point_ids": sorted(confirmed_ids),
            "intervals": [
                geo_svc._interval_view(iv) for iv in manifest["intervals"]
                if iv["status"] == "已确认"
            ],
            "spatial_coverage": coverage["spatial_coverage"],
            "source_points_file": points_path,
            "input_file": str(output_file),
        }

    def aggregate_for_verified_scope(self, batch_key: str, grid_level: str = "L6",
                                     time_bucket_sec: int = 3600) -> dict[str, Any]:
        """从已核验区间发起热力取数并交付，交付同时固化逐点输入凭据。"""
        prepared = self.prepare_verified_input(batch_key)
        if not prepared["verified_point_count"]:
            raise ValueError(
                f"批次 {batch_key} 没有已核验区间的观测点，无法发起热力取数"
            )
        points = self._read_jsonl(self._data_dir("heatmap_input") / f"{batch_key}_verified.jsonl")
        cells, total_points, peak_grid, peak_count = self._aggregate_points(
            batch_key, points, grid_level, time_bucket_sec
        )
        output_file = self._data_dir("heatmap") / f"{batch_key}_heatmap.jsonl"
        written = self._write_jsonl(output_file, cells)
        credential = self._write_credential(
            batch_key, points, cells, grid_level, time_bucket_sec, prepared
        )
        self._register_window(batch_key, credential, total_points, peak_grid, grid_level)
        return {
            "batch_key": batch_key,
            "grid_level": grid_level,
            "time_bucket_sec": time_bucket_sec,
            "heatmap_cells": written,
            "total_points_aggregated": total_points,
            "peak_grid": peak_grid,
            "peak_count": peak_count,
            "output_file": str(output_file),
            "credential_file": credential["credential_file"],
            "credential_id": credential["credential_id"],
            "verified_scope": {
                "intervals": prepared["intervals"],
                "point_ids": prepared["confirmed_point_ids"],
            },
        }

    # -- 输入凭据与原点复算 --

    def get_credential(self, batch_key: str) -> dict[str, Any]:
        """读取批次最近一次热力交付的输入凭据。"""
        index = self._read_checkpoint(f"heatmap_credential_index_{batch_key}")
        if index is None:
            raise FileNotFoundError(f"批次 {batch_key} 尚无热力交付凭据")
        return self._read_jsonl(self._data_dir("heatmap_credentials") /
                                f"{batch_key}_credential.jsonl")[-1]

    def read_credential_points(self, batch_key: str) -> list[dict[str, Any]]:
        """逐点读回凭据内嵌的输入点记录，供复算与核对。"""
        credential = self.get_credential(batch_key)
        return credential["points_snapshot"]

    def recompute_from_credential(self, batch_key: str) -> dict[str, Any]:
        """仅凭据内嵌快照重新计算热力统计，全程不读取当前点文件。

        原批次随后补点或修订（甚至点文件被删除）都不影响复算结果。
        """
        credential = self.get_credential(batch_key)
        cells, total_points, peak_grid, peak_count = self._aggregate_points(
            batch_key,
            credential["points_snapshot"],
            credential["grid_level"],
            credential["time_bucket_sec"],
        )
        delivered = credential["delivered_cells"]
        match = cells == delivered
        return {
            "batch_key": batch_key,
            "credential_id": credential["credential_id"],
            "recomputed_cells": cells,
            "recomputed_total": total_points,
            "delivered_total": credential["result_checksum"]["total_points"],
            "cell_count_match": len(cells) == len(delivered),
            "identical_to_delivery": match,
            "snapshot_checksum": credential["snapshot_checksum"],
            "result_checksum": credential["result_checksum"]["checksum"],
        }

    def reconcile_coverage(self, batch_key: str) -> dict[str, Any]:
        """首次热力读取前的覆盖核对：

        1. 这次可见的空间覆盖（当前全部实际观测）；
        2. 下一次还要补算的范围（已观测但未确认/未纳入凭据的区间）；
        3. 旧结果采用过哪些点（凭据内逐点点号）。
        """
        geo_svc = self._safe_import("app.services.geo_source", "GeoSourceService")()
        coverage = geo_svc.compute_coverage(batch_key)
        try:
            credential = self.get_credential(batch_key)
            used_point_ids = sorted(p["point_id"] for p in credential["points_snapshot"])
            credential_id = credential["credential_id"]
        except FileNotFoundError:
            used_point_ids = []
            credential_id = None

        used_set = set(used_point_ids)
        pending_intervals = []
        for iv in coverage["intervals"]:
            pending = [pid for pid in iv["observed_point_ids"] if pid not in used_set]
            if pending or iv["status"] != "已确认":
                pending_intervals.append({
                    "index": iv["index"],
                    "label": iv["label"],
                    "status": iv["status"],
                    "observed_point_ids": iv["observed_point_ids"],
                    "pending_point_ids": pending,
                })
        return {
            "batch_key": batch_key,
            "qc_round": coverage["qc_round"],
            "visible_spatial_coverage": coverage["spatial_coverage"],
            "visible_total": coverage["observed_total"],
            "confirmed_point_ids": coverage["confirmed_point_ids"],
            "next_pending_scope": pending_intervals,
            "delivered_credential_id": credential_id,
            "delivered_used_point_ids": used_point_ids,
        }

    # -- 兼容既有链路的全量聚合 --

    def aggregate_grid_heatmap(self, batch_key: str, grid_level: str = "L6",
                                time_bucket_sec: int = 3600) -> dict[str, Any]:
        """读取网格索引数据集，按网格层级和时间粒度进行热力聚合，生成热力聚合数据集文件。"""
        try:
            grid_svc = self._safe_import("app.services.grid_index", "GridIndexService")()
            grids, _ = grid_svc.get_grid_index(batch_key)
        except FileNotFoundError:
            grids_file = self._data_dir("grid_index") / f"{batch_key}_grid.jsonl"
            if not grids_file.exists():
                raise FileNotFoundError(f"批次 {batch_key} 的网格索引数据不存在，请先生成网格索引")
            grids = self._read_jsonl(grids_file)

        if not grids:
            return {"batch_key": batch_key, "heatmap_cells": 0}

        # 读取坐标转换后的定位点
        points_file = self._data_dir("coord_transformed") / f"{batch_key}_transformed.jsonl"
        if not points_file.exists():
            points_file = self._data_dir("geo_points") / f"{batch_key}.jsonl"
        points = self._read_jsonl(points_file) if points_file.exists() else []

        if not points:
            return {"batch_key": batch_key, "heatmap_cells": 0}

        cells, total_points, peak_grid, peak_count = self._aggregate_points(
            batch_key, points, grid_level, time_bucket_sec
        )
        output_file = self._data_dir("heatmap") / f"{batch_key}_heatmap.jsonl"
        written = self._write_jsonl(output_file, cells)

        return {
            "batch_key": batch_key,
            "grid_level": grid_level,
            "time_bucket_sec": time_bucket_sec,
            "heatmap_cells": written,
            "total_points_aggregated": total_points,
            "peak_grid": peak_grid,
            "peak_count": peak_count,
            "output_file": str(output_file),
        }

    def get_heatmap_data(self, batch_key: str) -> tuple[list[dict[str, Any]], str]:
        """读取指定批次的热力聚合数据集文件。"""
        file_path = self._data_dir("heatmap") / f"{batch_key}_heatmap.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"批次 {batch_key} 的热力聚合数据集不存在")
        data = self._read_jsonl(file_path)
        return data, str(file_path)

    # -- 内部辅助 --

    @staticmethod
    def _interval_of(manifest: dict[str, Any], point_id: Any) -> int | None:
        for iv in manifest["intervals"]:
            if point_id in iv["observed_point_ids"]:
                return iv["index"]
        return None

    @staticmethod
    def _point_hash(point: dict[str, Any]) -> str:
        payload = json_deterministic({k: v for k, v in point.items() if k != "point_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _aggregate_points(self, batch_key: str, points: list[dict[str, Any]],
                          grid_level: str, time_bucket_sec: int
                          ) -> tuple[list[dict[str, Any]], int, str | None, int]:
        """对给定点集合执行热力聚合；仅依赖入参，保证凭据复算与交付口径一致。"""
        heatmap: dict[str, dict[str, Any]] = {}
        for p in points:
            lat = float(p.get("trans_lat", p.get("lat", 0)))
            lon = float(p.get("trans_lon", p.get("lon", 0)))
            gh = self._geohash_encode(lat, lon, 6)
            ts = float(p.get("timestamp", 0))
            time_bucket = int(ts // time_bucket_sec) * time_bucket_sec
            key = f"{gh}_{time_bucket}"
            bucket = heatmap.setdefault(key, {
                "geohash": gh,
                "time_bucket": time_bucket,
                "point_count": 0,
                "lats": [],
                "lons": [],
            })
            bucket["point_count"] += 1
            bucket["lats"].append(lat)
            bucket["lons"].append(lon)

        cells = []
        peak_grid = None
        peak_count = 0
        total_points = 0
        for cell_id, key in enumerate(sorted(heatmap), start=1):
            info = heatmap[key]
            total_points += info["point_count"]
            center_lat = sum(info["lats"]) / len(info["lats"])
            center_lon = sum(info["lons"]) / len(info["lons"])
            cells.append({
                "cell_id": cell_id,
                "batch_key": batch_key,
                "geohash": info["geohash"],
                "grid_level": grid_level,
                "time_bucket": info["time_bucket"],
                "time_bucket_str": time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(info["time_bucket"])),
                "point_count": info["point_count"],
                "center_lat": round(center_lat, 6),
                "center_lon": round(center_lon, 6),
            })
            if info["point_count"] > peak_count:
                peak_count = info["point_count"]
                peak_grid = info["geohash"]
        return cells, total_points, peak_grid, peak_count

    def _write_credential(self, batch_key: str, points: list[dict[str, Any]],
                          cells: list[dict[str, Any]], grid_level: str,
                          time_bucket_sec: int, prepared: dict[str, Any]) -> dict[str, Any]:
        """固化逐点可读回的输入凭据：内嵌点记录、逐点哈希与整体快照校验值。"""
        snapshot = []
        for p in points:
            rec = dict(p)
            rec["point_hash"] = self._point_hash(rec)
            snapshot.append(rec)
        snapshot_checksum = self._compute_checksum(json_deterministic(snapshot))
        result_checksum = self._compute_checksum(json_deterministic(cells))
        credential = {
            "credential_id": f"HC-{batch_key}-{int(time.time())}",
            "batch_key": batch_key,
            "created_at": time.time(),
            "grid_level": grid_level,
            "time_bucket_sec": time_bucket_sec,
            "verified_intervals": prepared["intervals"],
            "spatial_coverage": prepared["spatial_coverage"],
            "points_snapshot": snapshot,
            "snapshot_checksum": snapshot_checksum,
            "point_count": len(snapshot),
            "result_checksum": {
                "checksum": result_checksum,
                "cell_count": len(cells),
                "total_points": sum(c["point_count"] for c in cells),
            },
            "delivered_cells": cells,
        }
        output_file = self._data_dir("heatmap_credentials") / f"{batch_key}_credential.jsonl"
        # 每个批次保留一份当前凭据：整份覆写，确保逐点读回的就是最近交付的输入
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json_deterministic(credential) + "\n")
        credential["credential_file"] = str(output_file)
        self._write_checkpoint(f"heatmap_credential_index_{batch_key}", {
            "batch_key": batch_key,
            "latest_credential_id": credential["credential_id"],
            "credential_file": str(output_file),
            "point_count": len(snapshot),
            "snapshot_checksum": snapshot_checksum,
            "updated_at": time.time(),
        })
        return credential

    def _register_window(self, batch_key: str, credential: dict[str, Any],
                         total_points: int, peak_grid: str | None,
                         grid_level: str) -> None:
        entry, _ = self.create_entry({
            "窗口编号": f"HM-{batch_key}-{credential['credential_id']}",
            "网格层级": grid_level,
            "时间粒度": f"{credential['time_bucket_sec']}s",
            "聚合指标": "point_count",
            "落点总量": str(total_points),
            "峰值网格": peak_grid or "",
            "窗口类型": "verified-scope",
            "聚合水位": f"已核验{total_points}点/凭据{credential['credential_id']}",
        })
        if entry:
            self.run_action(entry["id"], "确认产出")


def json_deterministic(obj: Any) -> str:
    """稳定序列化：键排序、不转义中文，保证凭据校验值可重复计算。"""
    import json
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
