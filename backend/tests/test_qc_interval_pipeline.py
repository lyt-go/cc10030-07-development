"""质检区间驱动的接入状态流转与热力凭据复算集成测试。

业务叙事：
- 首轮实收仅覆盖后半个质检区间，后段质检确认合格后热力先为这一范围准备输入；
- 前段两个观测迟到补齐，另一位质检员独立复验确认前段可用，后段确认持续有效；
- 正式接入不得混用未经确认的范围；
- 热力交付保留逐点可读回的输入凭据，批次补点/修订后统计仍能仅凭据复算；
- 首次热力读取前完成前段复验，并核对可见空间覆盖、待补算范围与旧结果采用过的点。
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def setup_data_dir(monkeypatch):
    """每个测试用例使用独立临时目录写入数据产物文件。"""
    tmpdir = Path(tempfile.mkdtemp(prefix="geo_qc_test_"))
    import app.services.base as base_mod
    original_root = base_mod.DATA_ROOT
    base_mod.DATA_ROOT = tmpdir
    for sub in ["geo_points", "coord_transformed", "track_segments", "dwell_events",
                "grid_index", "heatmap", "heatmap_input", "heatmap_credentials",
                "dedup_merged", "spatial_join", "poi_match", "region_aggregates",
                "od_matrix", "route_restore", "density_estimate", "quality_reports",
                "snapshots", "exports", "checkpoints"]:
        (tmpdir / sub).mkdir(parents=True, exist_ok=True)
    yield tmpdir
    base_mod.DATA_ROOT = original_root
    shutil.rmtree(tmpdir, ignore_errors=True)


# 固定时间轴：t0 ~ t0+8h 等宽切成两个质检区间，前段 0~4h，后段 4~8h
T0 = 1_800_000_000.0
HOUR = 3600.0

# 前后段使用相距足够远的坐标，保证 geohash(L6) 互不重叠
FRONT_COORD = (31.2300, 121.4700)  # 上海
REAR_COORD = (39.9042, 116.4074)   # 北京


def make_point(point_id: int, hour_offset: float, coord: tuple[float, float],
               **extra) -> dict:
    lat, lon = coord
    return {
        "point_id": point_id,
        "lat": lat,
        "lon": lon,
        "timestamp": T0 + hour_offset * HOUR,
        "accuracy": 10.0,
        "terminal": "TERM-QC1",
        **extra,
    }


@pytest.fixture()
def services():
    from app.services.geo_source import GeoSourceService
    from app.services.heatmap_aggregate import HeatmapAggregateService
    return GeoSourceService(), HeatmapAggregateService()


class TestQcIntervalFlow:
    """质检区间推进：待质检 → 可领取 →（撤回）待复验 →（独立复验）可领取。"""

    def test_full_interval_qc_and_heatmap_credential_story(self, services):
        geo, heatmap = services
        batch_key = "QC-FLOW-001"

        # 首轮实收：只有后段（4h~8h）的两个观测
        entry = geo.onboard_batch({
            "批次编号": batch_key,
            "采集终端": "TERM-QC1",
            "定位方式": "GPS",
            "采样频率": "1Hz",
        }, point_count=0)
        batch_id = entry["id"]
        geo.observe_points(batch_key, [
            make_point(1, 5.0, REAR_COORD),
            make_point(2, 6.5, REAR_COORD),
        ])

        # 提交质检并按固定边界初始化两个区间
        geo.submit_for_qc(batch_id)
        manifest = geo.init_qc_intervals(batch_key, interval_count=2,
                                         bounds=(T0, T0 + 8 * HOUR))
        assert [iv["label"] for iv in manifest["intervals"]] == ["区间1", "区间2"]

        coverage = geo.compute_coverage(batch_key)
        front, rear = coverage["intervals"]
        # 实际观测只覆盖后段
        assert front["observed_point_ids"] == []
        assert rear["observed_point_ids"] == [1, 2]
        assert front["status"] == "待质检" and rear["status"] == "待质检"
        assert coverage["confirmed_point_ids"] == []

        # 后段质检确认合格 → 批次进入可领取，前段仍待质检
        geo.confirm_interval(batch_key, 1, inspector="qc_zhang")
        entry_now = geo.get_entry(batch_id)
        assert entry_now["status"] == "可领取"
        manifest = geo.get_qc_manifest(batch_key)
        assert manifest["intervals"][1]["status"] == "已确认"
        assert manifest["intervals"][1]["confirmed_by"] == "qc_zhang"

        # 热力只为已核验的后段范围准备输入
        prepared = heatmap.prepare_verified_input(batch_key)
        assert prepared["verified_point_count"] == 2
        assert sorted(prepared["confirmed_point_ids"]) == [1, 2]
        rear_geohashes = prepared["spatial_coverage"]["geohashes"]

        # 正式接入不得混用未经确认的范围：请求全区间接入应被拒绝
        with pytest.raises(ValueError, match="未经确认"):
            geo.formal_onboard(batch_id, interval_indices=[0, 1])
        with pytest.raises(ValueError, match="未经确认"):
            geo.formal_onboard(batch_id)  # 全部已观测区间里前段尚未确认

        # 首次热力读取（交付）：仅后段两点，并固化输入凭据
        delivery = heatmap.aggregate_for_verified_scope(batch_key)
        assert delivery["total_points_aggregated"] == 2
        assert delivery["verified_scope"]["point_ids"] == [1, 2]
        credential = heatmap.get_credential(batch_key)
        assert credential["point_count"] == 2
        # 凭据逐点可读回，且每点带哈希
        readback = heatmap.read_credential_points(batch_key)
        assert sorted(p["point_id"] for p in readback) == [1, 2]
        assert all(len(p["point_hash"]) == 64 for p in readback)
        delivered_cells = credential["delivered_cells"]

        # 前段两个观测迟到补齐：开启新一轮质检，前段进入待复验；后段确认继续有效
        arrival = geo.observe_points(batch_key, [
            make_point(3, 1.0, FRONT_COORD, is_backfill=True),
            make_point(4, 2.5, FRONT_COORD, is_backfill=True),
        ])
        assert arrival["qc_round"] == 2
        manifest = geo.get_qc_manifest(batch_key)
        assert manifest["round"] == 2
        assert manifest["intervals"][0]["status"] == "待复验"
        assert manifest["intervals"][1]["status"] == "已确认"  # 后段确认持续有效
        assert geo.get_entry(batch_id)["status"] == "待复验"

        # 同一位质检员不能复验自己批次的区间，必须另一位质检员独立复验
        with pytest.raises(ValueError, match="另一位质检员"):
            geo.reverify_interval(batch_key, 0, inspector="qc_zhang")
        geo.reverify_interval(batch_key, 0, inspector="qc_li")
        manifest = geo.get_qc_manifest(batch_key)
        assert manifest["intervals"][0]["status"] == "已确认"
        assert manifest["intervals"][0]["confirmed_by"] == "qc_li"
        assert geo.get_entry(batch_id)["status"] == "可领取"

        # 首次热力读取前的覆盖核对
        reconciliation = heatmap.reconcile_coverage(batch_key)
        # 可见空间覆盖同时包含前后段两套不相交网格
        all_geohashes = reconciliation["visible_spatial_coverage"]["geohashes"]
        assert set(rear_geohashes).issubset(set(all_geohashes))
        assert len(all_geohashes) > len(rear_geohashes)
        # 下一次还要补算的范围：前段两个新点尚未进入任何凭据
        pending = reconciliation["next_pending_scope"]
        pending_ids = {pid for iv in pending for pid in iv["pending_point_ids"]}
        assert pending_ids == {3, 4}
        # 旧结果采用过哪些点：仅后段 1、2
        assert reconciliation["delivered_used_point_ids"] == [1, 2]

        # 原批次补点/修订（追加新观测、篡改当前点文件）后，旧统计仍能仅凭据复算
        geo.observe_points(batch_key, [make_point(5, 3.2, FRONT_COORD)])
        points_file = Path(geo.get_batch_points(batch_key)[1])
        original_file = points_file.read_text(encoding="utf-8")
        points_file.write_text(original_file + '{"point_id": 99, "tampered": true}\n',
                               encoding="utf-8")
        recomputed = heatmap.recompute_from_credential(batch_key)
        assert recomputed["identical_to_delivery"] is True
        assert recomputed["recomputed_total"] == 2
        assert recomputed["recomputed_cells"] == delivered_cells
        # 即便点文件被删除，凭据复算依然成立
        points_file.unlink()
        recomputed_again = heatmap.recompute_from_credential(batch_key)
        assert recomputed_again["identical_to_delivery"] is True
        # 恢复点文件供后续接入流程使用（证明复算阶段完全不依赖它）
        points_file.write_text(original_file, encoding="utf-8")

        # 全区间已核验，正式接入成功，且只接入已确认范围
        result = geo.formal_onboard(batch_id)
        assert result["batch_entry"]["status"] == "已接入"
        assert {iv["index"] for iv in result["onboard_scope"]} == {0, 1}
        assert "coord_transform" in result["auto_triggers"]

    def test_withdraw_confirmation_returns_to_recheck(self, services):
        geo, heatmap = services
        batch_key = "QC-WITHDRAW-001"
        entry = geo.onboard_batch({
            "批次编号": batch_key,
            "采集终端": "TERM-QC2",
            "定位方式": "GPS",
        }, point_count=0)
        geo.observe_points(batch_key, [
            make_point(1, 5.0, REAR_COORD),
            make_point(2, 6.0, REAR_COORD),
        ])
        geo.submit_for_qc(entry["id"])
        geo.init_qc_intervals(batch_key, 2, (T0, T0 + 8 * HOUR))
        geo.confirm_interval(batch_key, 1, inspector="qc_zhang")
        assert geo.get_entry(entry["id"])["status"] == "可领取"

        # 撤回确认 → 后段回到待复验，批次整体回到待复验
        view = geo.withdraw_confirmation(batch_key, 1, inspector="qc_zhang")
        assert view["status"] == "待复验"
        assert geo.get_entry(entry["id"])["status"] == "待复验"

        # 原质检员不能复验，换人独立复验后重新可领取
        with pytest.raises(ValueError, match="另一位质检员"):
            geo.reverify_interval(batch_key, 1, inspector="qc_zhang")
        geo.reverify_interval(batch_key, 1, inspector="qc_li")
        assert geo.get_entry(entry["id"])["status"] == "可领取"

        # 撤回状态下热力取数没有已核验输入，应被拒绝
        geo.withdraw_confirmation(batch_key, 1, inspector="qc_li")
        with pytest.raises(ValueError, match="没有已核验区间"):
            heatmap.aggregate_for_verified_scope(batch_key)

    def test_coverage_follows_actual_observation(self, services):
        geo, _ = services
        batch_key = "QC-COVERAGE-001"
        entry = geo.onboard_batch({
            "批次编号": batch_key,
            "采集终端": "TERM-QC3",
            "定位方式": "GPS",
        }, point_count=0)
        geo.observe_points(batch_key, [make_point(1, 0.5, FRONT_COORD)])
        geo.submit_for_qc(entry["id"])
        geo.init_qc_intervals(batch_key, 2, (T0, T0 + 8 * HOUR))

        coverage = geo.compute_coverage(batch_key)
        # 实际观测落在前段；后段无观测即不可接入
        assert coverage["intervals"][0]["observed_point_ids"] == [1]
        assert coverage["intervals"][1]["observed_point_ids"] == []
        bbox = coverage["spatial_coverage"]["bbox"]
        assert bbox["min_lat"] == FRONT_COORD[0]

        # 无确认区间时不能正式接入
        with pytest.raises(ValueError, match="未经确认"):
            geo.formal_onboard(entry["id"])

    def test_legacy_confirm_batch_remains_compatible(self, services):
        geo, _ = services
        batch_key = "QC-LEGACY-001"
        entry = geo.onboard_batch({
            "批次编号": batch_key,
            "采集终端": "TERM-QC4",
            "定位方式": "GPS",
            "采样频率": "1Hz",
        }, point_count=12)
        result = geo.confirm_batch(entry["id"])
        assert result["batch_key"] == batch_key
        assert result["batch_entry"]["status"] == "已接入"
        assert result["auto_triggers"]["coord_transform"]["transformed_points"] == 12
        manifest = geo.get_qc_manifest(batch_key)
        assert all(iv["status"] == "已确认" for iv in manifest["intervals"])
