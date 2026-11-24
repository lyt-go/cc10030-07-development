"""定位源接入领域服务：采集定位点数据批次登记、采样频率与采集水位推进。

在基础采集能力之上，新增按质检区间推进可用状态的流程：

- 接入状态沿 ``草稿 → 待质检 → 可领取 → 待复验 → 已接入`` 推进，撤回确认后回到
  ``待复验``，复验须由另一位质检员独立完成。
- 可接入范围按实际观测覆盖计算：观测点按时间戳归属到稳定的质检区间，未观测或
  未核验的区间不得混入正式接入与热力取数。
- 迟到观测开启新一轮质检，先前区间的确认结论继续有效。
"""
from __future__ import annotations

import json
import time
from typing import Any

from app.services.base import BaseModuleService
from app.store import store

# 质检区间的三种状态
INTERVAL_PENDING_QC = "待质检"
INTERVAL_CONFIRMED = "已确认"
INTERVAL_RECHECK = "待复验"


class GeoSourceService(BaseModuleService):
    MODULE = "geo_source"
    ENTITY = "采集批次"
    OPERATION_TERM = "采集"
    LIST_FIELDS = ["批次编号", "采集终端", "定位方式", "采样频率", "记录条数", "时间跨度", "采集水位", "批次校验值"]
    REQUIRED_FIELDS = ["批次编号", "采集终端", "定位方式"]
    STATUS_ORDER = ["草稿", "待质检", "可领取", "待复验", "已接入", "已拒收"]
    ACTION_RULES = {
        "提交质检": "待质检",
        "确认可领取": "可领取",
        "撤回确认": "待复验",
        "复验通过": "可领取",
        "确认接入": "已接入",
        "标记拒收": "已拒收",
    }
    NEGATIVE_ACTIONS = []

    # -- 批次登记与采集 --

    def onboard_batch(self, values: dict[str, Any], point_count: int = 50) -> dict[str, Any]:
        """接入新的定位点采集批次，生成模拟定位点序列并写入定位点数据集文件。"""
        entry, missing = self.create_entry(values)
        if entry is None:
            raise ValueError(f"缺少必填字段: {missing}")
        batch_key = entry["批次编号"]
        # 写入采集位点
        self._write_checkpoint(f"geo_collect_{batch_key}", {
            "batch_key": batch_key,
            "offset": 0,
            "point_count": point_count,
            "status": "active",
            "created_at": time.time(),
        })
        # 生成模拟定位点并写入定位数据文件
        points = self._generate_sample_points(count=point_count)
        for p in points:
            p["batch_key"] = batch_key
            p["terminal"] = entry.get("采集终端", "UNKNOWN")
        output_file = self._data_dir("geo_points") / f"{batch_key}.jsonl"
        written = self._write_jsonl(output_file, points)
        # 更新记录条数
        entry["记录条数"] = str(written)
        if points:
            first_ts = min(p["timestamp"] for p in points)
            last_ts = max(p["timestamp"] for p in points)
            entry["时间跨度"] = f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(first_ts))} ~ {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(last_ts))}"
            checksum = self._compute_checksum(json.dumps(points, default=str))
        else:
            entry["时间跨度"] = ""
            checksum = self._compute_checksum("")
        entry["批次校验值"] = checksum
        entry["采集水位"] = f"已采集{written}条"
        return entry

    def observe_points(self, batch_key: str, points: list[dict[str, Any]]) -> dict[str, Any]:
        """登记一批实际观测（含迟到补齐观测），追加写入批次定位点数据集。

        若批次已有区间被确认过，新观测开启下一轮质检：新覆盖到的区间进入
        ``待复验``，等待另一位质检员独立复验；已确认区间保持原结论不变。
        """
        output_file = self._data_dir("geo_points") / f"{batch_key}.jsonl"
        existing = self._read_jsonl(output_file)
        next_id = max((int(p.get("point_id", 0)) for p in existing), default=0) + 1
        normalized: list[dict[str, Any]] = []
        for p in points:
            rec = dict(p)
            rec.setdefault("point_id", next_id)
            next_id = max(next_id, int(rec["point_id"]) + 1)
            rec["batch_key"] = batch_key
            rec.setdefault("terminal", "UNKNOWN")
            normalized.append(rec)
        written = self._write_jsonl(output_file, normalized)

        manifest = self._read_checkpoint(f"qc_manifest_{batch_key}")
        if manifest is not None:
            had_confirmation = any(
                iv["status"] == INTERVAL_CONFIRMED for iv in manifest["intervals"]
            )
            self._refresh_observed(batch_key, manifest)
            if had_confirmation:
                # 迟到观测触发新一轮独立复验，只影响尚未确认的区间
                manifest["round"] = int(manifest.get("round", 1)) + 1
                for iv in manifest["intervals"]:
                    if iv["observed_point_ids"] and iv["status"] == INTERVAL_PENDING_QC:
                        iv["status"] = INTERVAL_RECHECK
                        iv["history"].append({
                            "action": "迟到观测转入复验",
                            "inspector": None,
                            "at": time.time(),
                        })
            self._save_manifest(manifest)
            self._sync_batch_status(batch_key, manifest)
        return {
            "batch_key": batch_key,
            "appended_points": written,
            "total_points": len(existing) + written,
            "qc_round": manifest["round"] if manifest else None,
        }

    # -- 质检区间清单 --

    def init_qc_intervals(self, batch_key: str, interval_count: int = 2,
                          bounds: tuple[float, float] | None = None) -> dict[str, Any]:
        """初始化质检区间清单。

        - 显式给定 ``bounds=(start_ts, end_ts)`` 时按等宽切分，边界在后续迟到
          观测中保持稳定；
        - 未给定时按当前实际观测的时间跨度等宽切分。
        """
        if interval_count < 1:
            raise ValueError("质检区间数量必须大于0")
        if bounds is None:
            points, _ = self.get_batch_points(batch_key)
            if not points:
                raise ValueError(f"批次 {batch_key} 尚无观测点，无法按实际观测计算区间边界")
            timestamps = [float(p["timestamp"]) for p in points]
            start_ts, end_ts = min(timestamps), max(timestamps)
            if end_ts <= start_ts:
                # 单点或同刻观测：给出一秒的最小有效跨度，避免零宽区间
                end_ts = start_ts + 1.0
            bounds = (start_ts, end_ts)
        start_ts, end_ts = float(bounds[0]), float(bounds[1])
        if end_ts <= start_ts:
            raise ValueError("质检区间结束时间必须晚于开始时间")
        width = (end_ts - start_ts) / interval_count
        intervals = []
        for i in range(interval_count):
            iv_start = start_ts + i * width
            iv_end = end_ts if i == interval_count - 1 else start_ts + (i + 1) * width
            intervals.append({
                "index": i,
                "label": f"区间{i + 1}",
                "start_ts": iv_start,
                "end_ts": iv_end,
                "status": INTERVAL_PENDING_QC,
                "observed_point_ids": [],
                "confirmed_by": None,
                "confirmed_at": None,
                "history": [],
            })
        manifest = {
            "batch_key": batch_key,
            "interval_count": interval_count,
            "round": 1,
            "intervals": intervals,
            "round_inspectors": {},
        }
        self._refresh_observed(batch_key, manifest)
        self._save_manifest(manifest)
        return manifest

    def get_qc_manifest(self, batch_key: str) -> dict[str, Any]:
        """读取批次质检区间清单，不存在时抛出异常。"""
        manifest = self._read_checkpoint(f"qc_manifest_{batch_key}")
        if manifest is None:
            raise FileNotFoundError(f"批次 {batch_key} 的质检区间清单尚未初始化")
        return manifest

    def submit_for_qc(self, batch_id: int) -> dict[str, Any]:
        """提交质检：批次进入待质检，并确保质检区间清单已按实际观测初始化。"""
        entry = self._find_entry_by_id(batch_id)
        batch_key = entry["批次编号"]
        if self._read_checkpoint(f"qc_manifest_{batch_key}") is None:
            self.init_qc_intervals(batch_key)
        entry, msg = self.run_action(batch_id, "提交质检")
        return {"entry": entry, "message": msg, "manifest": self.get_qc_manifest(batch_key)}

    def compute_coverage(self, batch_key: str) -> dict[str, Any]:
        """按实际观测覆盖计算可接入范围：逐点归属质检区间并汇总空间覆盖。"""
        manifest = self.get_qc_manifest(batch_key)
        self._refresh_observed(batch_key, manifest)
        self._save_manifest(manifest)
        points, _ = self.get_batch_points(batch_key)
        return self._coverage_summary(batch_key, manifest, points)

    def confirm_interval(self, batch_key: str, interval_index: int,
                         inspector: str) -> dict[str, Any]:
        """质检员确认指定区间合格，区间进入已确认；批次据此推进到可领取。"""
        manifest = self.get_qc_manifest(batch_key)
        iv = self._interval_at(manifest, interval_index)
        if not iv["observed_point_ids"]:
            raise ValueError(f"{iv['label']} 尚无实际观测，不能确认合格")
        if iv["status"] == INTERVAL_RECHECK:
            raise ValueError(f"{iv['label']} 处于待复验，请使用复验流程")
        self._mark_confirmed(iv, inspector, "质检确认")
        manifest.setdefault("round_inspectors", {}).setdefault(str(manifest["round"]), [])
        if inspector not in manifest["round_inspectors"][str(manifest["round"])]:
            manifest["round_inspectors"][str(manifest["round"])].append(inspector)
        self._save_manifest(manifest)
        self._sync_batch_status(batch_key, manifest)
        return self._interval_view(iv)

    def withdraw_confirmation(self, batch_key: str, interval_index: int,
                              inspector: str) -> dict[str, Any]:
        """撤回已确认区间的合格结论，该区间回到待复验，批次同步回到待复验。"""
        manifest = self.get_qc_manifest(batch_key)
        iv = self._interval_at(manifest, interval_index)
        if iv["status"] != INTERVAL_CONFIRMED:
            raise ValueError(f"{iv['label']} 当前为{iv['status']}，无可撤回的确认")
        iv["status"] = INTERVAL_RECHECK
        iv["history"].append({
            "action": "撤回确认",
            "inspector": inspector,
            "at": time.time(),
        })
        self._save_manifest(manifest)
        self._sync_batch_status(batch_key, manifest)
        return self._interval_view(iv)

    def reverify_interval(self, batch_key: str, interval_index: int,
                          inspector: str) -> dict[str, Any]:
        """独立复验：待复验区间须由另一位质检员（不同于原确认人）复验通过。"""
        manifest = self.get_qc_manifest(batch_key)
        iv = self._interval_at(manifest, interval_index)
        if iv["status"] != INTERVAL_RECHECK:
            raise ValueError(f"{iv['label']} 当前为{iv['status']}，不在待复验状态")
        # 独立复验：复验人不能是此前在该批次任何区间做过确认的质检员
        prior_inspectors = {
            h.get("inspector")
            for other in manifest["intervals"]
            for h in other.get("history", [])
            if h.get("inspector")
        }
        if inspector in prior_inspectors:
            raise ValueError(
                f"{iv['label']} 的复验须由另一位质检员独立完成，{inspector} 已参与过该批次质检确认"
            )
        self._mark_confirmed(iv, inspector, "独立复验通过")
        manifest.setdefault("round_inspectors", {}).setdefault(str(manifest["round"]), [])
        if inspector not in manifest["round_inspectors"][str(manifest["round"])]:
            manifest["round_inspectors"][str(manifest["round"])].append(inspector)
        self._save_manifest(manifest)
        self._sync_batch_status(batch_key, manifest)
        return self._interval_view(iv)

    def formal_onboard(self, batch_id: int,
                       interval_indices: list[int] | None = None) -> dict[str, Any]:
        """正式接入：只允许接入已核验区间，拒绝混用未经确认的范围。

        接入成功后沿用既有下游链路（坐标标准化、点位归并、网格索引）。
        """
        entry = self._find_entry_by_id(batch_id)
        batch_key = entry["批次编号"]
        manifest = self.get_qc_manifest(batch_key)
        self._refresh_observed(batch_key, manifest)
        if not any(iv["observed_point_ids"] for iv in manifest["intervals"]):
            raise ValueError(f"批次 {batch_key} 无实际观测覆盖，不能正式接入")
        # 默认范围为全部质检区间：未观测或未核验的区间都会拒绝接入，杜绝混用
        scope = manifest["intervals"] if interval_indices is None else [
            self._interval_at(manifest, idx) for idx in interval_indices
        ]
        unconfirmed = [iv["label"] for iv in scope if iv["status"] != INTERVAL_CONFIRMED]
        if unconfirmed:
            raise ValueError(
                f"批次 {batch_key} 正式接入拒绝混用未经确认的范围：{'、'.join(unconfirmed)}"
            )
        entry, msg = self.run_action(batch_id, "确认接入")
        triggers = self._run_downstream_triggers(batch_key)
        manifest["onboard_scope"] = [iv["index"] for iv in scope]
        manifest["onboard_at"] = time.time()
        self._save_manifest(manifest)
        return {
            "batch_entry": entry,
            "batch_key": batch_key,
            "message": msg,
            "onboard_scope": [self._interval_view(iv) for iv in scope],
            "confirmed_at": time.time(),
            "auto_triggers": triggers,
        }

    def confirm_batch(self, batch_id: int) -> dict[str, Any]:
        """确认接入采集批次（兼容链路）：自动核验全部已观测区间后执行正式接入。

        自动链路以 ``system-auto`` 身份完成质检确认，严格的人工质检流程请使用
        :meth:`submit_for_qc` / :meth:`confirm_interval` / :meth:`formal_onboard`。
        """
        entry = self._find_entry_by_id(batch_id)
        batch_key = entry["批次编号"]
        if self._read_checkpoint(f"qc_manifest_{batch_key}") is None:
            self.init_qc_intervals(batch_key)
        manifest = self.get_qc_manifest(batch_key)
        self._refresh_observed(batch_key, manifest)
        for iv in manifest["intervals"]:
            if iv["observed_point_ids"] and iv["status"] != INTERVAL_CONFIRMED:
                self._mark_confirmed(iv, "system-auto", "系统自动确认")
        self._save_manifest(manifest)
        self._sync_batch_status(batch_key, manifest)

        onboard = self.formal_onboard(batch_id)
        return {
            "batch_entry": onboard["batch_entry"],
            "batch_key": batch_key,
            "confirmed_at": onboard["confirmed_at"],
            "auto_triggers": onboard["auto_triggers"],
        }

    # -- 数据读取 --

    def get_batch_points(self, batch_key: str) -> tuple[list[dict[str, Any]], str]:
        """读取指定采集批次的定位点数据文件，返回定位点记录列表和文件路径。"""
        file_path = self._data_dir("geo_points") / f"{batch_key}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"采集批次 {batch_key} 的定位点数据文件不存在")
        points = self._read_jsonl(file_path)
        return points, str(file_path)

    def list_active_batches(self) -> list[dict[str, Any]]:
        """获取所有已接入的活跃采集批次列表。"""
        rows = store.rows(self.MODULE)
        return [r for r in rows if r.get("status") == "已接入"]

    # -- 内部辅助 --

    def _find_entry_by_id(self, batch_id: int) -> dict[str, Any]:
        entry = store.find(self.MODULE, batch_id)
        if entry is None:
            raise ValueError(f"{self.ENTITY} {batch_id} 不存在或已归档")
        return entry

    def _find_entry_by_key(self, batch_key: str) -> dict[str, Any] | None:
        for row in store.rows(self.MODULE):
            if row.get("批次编号") == batch_key:
                return row
        return None

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self._write_checkpoint(f"qc_manifest_{manifest['batch_key']}", manifest)

    @staticmethod
    def _interval_at(manifest: dict[str, Any], interval_index: int) -> dict[str, Any]:
        for iv in manifest["intervals"]:
            if iv["index"] == interval_index:
                return iv
        raise ValueError(f"质检区间 {interval_index} 不存在")

    @staticmethod
    def _mark_confirmed(iv: dict[str, Any], inspector: str, action: str) -> None:
        iv["status"] = INTERVAL_CONFIRMED
        iv["confirmed_by"] = inspector
        iv["confirmed_at"] = time.time()
        iv["history"].append({"action": action, "inspector": inspector, "at": time.time()})

    def _locate_interval(self, manifest: dict[str, Any], ts: float) -> dict[str, Any]:
        """按时间戳归属区间；越界观测归入最近区间并扩展该区间的实际边界。"""
        intervals = manifest["intervals"]
        for iv in intervals:
            if iv["start_ts"] <= ts < iv["end_ts"]:
                return iv
        target = intervals[0] if ts < intervals[0]["start_ts"] else intervals[-1]
        if ts < target["start_ts"]:
            target["start_ts"] = ts
        elif ts >= target["end_ts"]:
            target["end_ts"] = ts + 1e-6
        return target

    def _refresh_observed(self, batch_key: str, manifest: dict[str, Any]) -> None:
        """重新读取当前点文件，把实际观测逐点归属到质检区间。"""
        points_file = self._data_dir("geo_points") / f"{batch_key}.jsonl"
        points = self._read_jsonl(points_file) if points_file.exists() else []
        for iv in manifest["intervals"]:
            iv["observed_point_ids"] = []
        for p in points:
            iv = self._locate_interval(manifest, float(p["timestamp"]))
            iv["observed_point_ids"].append(p.get("point_id"))

    def _sync_batch_status(self, batch_key: str, manifest: dict[str, Any]) -> None:
        """根据区间状态推导批次整体可用状态并写回资源记录。"""
        entry = self._find_entry_by_key(batch_key)
        if entry is None:
            return
        if any(iv["status"] == INTERVAL_RECHECK for iv in manifest["intervals"]):
            entry["status"] = "待复验"
        elif any(iv["status"] == INTERVAL_CONFIRMED for iv in manifest["intervals"]):
            entry["status"] = "可领取"
        elif any(iv["observed_point_ids"] for iv in manifest["intervals"]):
            entry["status"] = "待质检"
        entry["pending"] = entry["status"] != self.STATUS_ORDER[-1]

    def _coverage_summary(self, batch_key: str, manifest: dict[str, Any],
                          points: list[dict[str, Any]]) -> dict[str, Any]:
        point_index = {p.get("point_id"): p for p in points}
        interval_views = []
        for iv in manifest["intervals"]:
            iv_points = [point_index[pid] for pid in iv["observed_point_ids"]
                         if pid in point_index]
            geohashes = sorted({
                self._geohash_encode(float(p["lat"]), float(p["lon"]), 6)
                for p in iv_points
            })
            interval_views.append({
                **self._interval_view(iv),
                "geohashes": geohashes,
                "point_count": len(iv_points),
            })
        geohashes = sorted({
            self._geohash_encode(float(p["lat"]), float(p["lon"]), 6) for p in points
        })
        bbox = None
        if points:
            lats = [float(p["lat"]) for p in points]
            lons = [float(p["lon"]) for p in points]
            bbox = {"min_lat": min(lats), "max_lat": max(lats),
                    "min_lon": min(lons), "max_lon": max(lons)}
        return {
            "batch_key": batch_key,
            "qc_round": manifest.get("round", 1),
            "observed_total": len(points),
            "spatial_coverage": {"geohashes": geohashes, "bbox": bbox},
            "intervals": interval_views,
            "confirmed_point_ids": [
                pid for iv in manifest["intervals"] if iv["status"] == INTERVAL_CONFIRMED
                for pid in iv["observed_point_ids"]
            ],
        }

    @staticmethod
    def _interval_view(iv: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": iv["index"],
            "label": iv["label"],
            "start_ts": iv["start_ts"],
            "end_ts": iv["end_ts"],
            "status": iv["status"],
            "observed_point_ids": list(iv["observed_point_ids"]),
            "confirmed_by": iv.get("confirmed_by"),
            "confirmed_at": iv.get("confirmed_at"),
        }

    def _run_downstream_triggers(self, batch_key: str) -> dict[str, Any]:
        """正式接入后触发坐标标准化、点位归并与网格索引生成。"""
        triggers: dict[str, Any] = {}
        try:
            coord_svc = self._safe_import("app.services.coord_transform", "CoordTransformService")()
            triggers["coord_transform"] = coord_svc.transform_batch(batch_key)
        except Exception as e:
            triggers["coord_transform"] = {"error": str(e)}

        try:
            dedup_svc = self._safe_import("app.services.dedup_merge", "DedupMergeService")()
            triggers["dedup_merge"] = dedup_svc.dedup_batch(batch_key)
        except Exception as e:
            triggers["dedup_merge"] = {"error": str(e)}

        try:
            grid_svc = self._safe_import("app.services.grid_index", "GridIndexService")()
            triggers["grid_index"] = grid_svc.index_points_from_batch(batch_key)
        except Exception as e:
            triggers["grid_index"] = {"error": str(e)}
        return triggers
