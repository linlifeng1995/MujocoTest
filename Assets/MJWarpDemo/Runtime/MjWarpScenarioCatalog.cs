using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.SceneManagement;

namespace MJWarpDemo
{
    [Serializable]
    public sealed class MjWarpScenarioInfo
    {
        public string SceneName;
        public string ScenarioId;
        public string DisplayName;
        public string BusinessType;
        public string Description;

        public MjWarpScenarioInfo(
            string sceneName,
            string scenarioId,
            string displayName,
            string businessType,
            string description)
        {
            SceneName = sceneName;
            ScenarioId = scenarioId;
            DisplayName = displayName;
            BusinessType = businessType;
            Description = description;
        }
    }

    public static class MjWarpScenarioCatalog
    {
        private static readonly IReadOnlyList<MjWarpScenarioInfo> scenarios = new[]
        {
            new MjWarpScenarioInfo(
                "PlanarPushScene",
                "planar_push",
                "物流推运",
                "仓储与非抓取式操作",
                "机械臂把散件推送到指定收货区域。"),
            new MjWarpScenarioInfo(
                "PrecisionAssemblyScene",
                "precision_insert",
                "精密装配",
                "制造业装配与插入",
                "机械臂把方形定位块推入带导向槽的装配工位。"),
            new MjWarpScenarioInfo(
                "QualityInspectionScene",
                "quality_inspection",
                "质量检测到位",
                "工业质检与多工位巡检",
                "机械臂末端到达随机检测工位并稳定保持。"),
            new MjWarpScenarioInfo(
                "WarehouseNavigationScene",
                "warehouse_navigation",
                "仓储移动机器人",
                "AMR 路径规划与避障",
                "移动底盘穿过货架通道到达随机库位。"),
            new MjWarpScenarioInfo(
                "PandaPickPlaceScene",
                "panda_pick_place",
                "Panda 抓取放置",
                "标准单臂抓放与数据交付",
                "Franka Panda 抓取随机方块并放入目标料盒。"),
            new MjWarpScenarioInfo(
                "PandaPegInsertScene",
                "panda_peg_insert",
                "Panda 精密插入",
                "接触型装配与插入",
                "Franka Panda 抓取带法兰定位销并插入随机孔位。"),
        };

        public static IReadOnlyList<MjWarpScenarioInfo> All => scenarios;

        public static MjWarpScenarioInfo Active
        {
            get
            {
                string sceneName = SceneManager.GetActiveScene().name;
                return scenarios.FirstOrDefault(item =>
                           string.Equals(item.SceneName, sceneName, StringComparison.OrdinalIgnoreCase))
                       ?? scenarios[0];
            }
        }

        public static MjWarpScenarioInfo FindById(string scenarioId)
        {
            return scenarios.FirstOrDefault(item =>
                       string.Equals(item.ScenarioId, scenarioId, StringComparison.OrdinalIgnoreCase))
                   ?? scenarios[0];
        }
    }
}
