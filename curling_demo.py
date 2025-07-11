#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冰壶比赛规则演示程序
展示用户描述的冰壶比赛规则的实现效果
"""

from curling_rules_manager import CurlingRulesManager

def demonstrate_scoring_rules():
    """演示得分规则"""
    print("=" * 50)
    print("演示得分规则：位于大本营中且距离圆心更近的冰壶得分")
    print("=" * 50)
    
    rules_manager = CurlingRulesManager()
    
    # 场景1：Team1有更接近圆心的冰壶
    print("\n场景1：Team1有冰壶更接近圆心")
    team1_stones = [
        [2.375, 4.88],      # 正中心
        [2.5, 5.0],         # 稍偏离
        None, None, None, None, None, None
    ]
    team2_stones = [
        [2.6, 5.2],         # 更远
        [2.7, 5.3],         # 更远
        None, None, None, None, None, None
    ]
    
    team1_score, team2_score = rules_manager.calculate_end_score(team1_stones, team2_stones)
    print(f"Team1得分: {team1_score}, Team2得分: {team2_score}")
    
    # 场景2：只有一队有冰壶在大本营内
    print("\n场景2：只有Team2有冰壶在大本营内")
    team1_stones = [None] * 8  # 没有冰壶在大本营内
    team2_stones = [
        [2.4, 4.9],         # 在大本营内
        [2.6, 5.1],         # 在大本营内
        [2.8, 5.3],         # 在大本营内
        None, None, None, None, None
    ]
    
    team1_score, team2_score = rules_manager.calculate_end_score(team1_stones, team2_stones)
    print(f"Team1得分: {team1_score}, Team2得分: {team2_score}")
    
    # 场景3：平局情况
    print("\n场景3：两队都没有冰壶在大本营内")
    team1_stones = [None] * 8
    team2_stones = [None] * 8
    
    team1_score, team2_score = rules_manager.calculate_end_score(team1_stones, team2_stones)
    print(f"Team1得分: {team1_score}, Team2得分: {team2_score}")

def demonstrate_first_hand_rules():
    """演示先后手规则"""
    print("\n" + "=" * 50)
    print("演示先后手规则：得分队伍下一局为先手，平局则交换先后手")
    print("=" * 50)
    
    rules_manager = CurlingRulesManager()
    rules_manager.first_hand_team = 'team1'  # 假设第一局team1先手
    
    print(f"第1局先手: {rules_manager.first_hand_team}")
    
    # 模拟几局比赛
    scenarios = [
        (2, 1, "Team1获胜"),
        (0, 0, "平局"),
        (1, 3, "Team2获胜"),
        (0, 0, "平局"),
        (1, 0, "Team1获胜")
    ]
    
    for i, (team1_score, team2_score, description) in enumerate(scenarios, 1):
        print(f"\n第{i}局结果: {description} (Team1: {team1_score}, Team2: {team2_score})")
        
        # 模拟冰壶位置（简化）
        team1_stones = [[2.4, 4.9]] * team1_score + [None] * (8 - team1_score)
        team2_stones = [[2.5, 5.0]] * team2_score + [None] * (8 - team2_score)
        
        rules_manager.update_game_state(team1_stones, team2_stones)
        
        if not rules_manager.is_game_over():
            print(f"下一局(第{i+1}局)先手: {rules_manager.first_hand_team}")
        else:
            print("比赛结束")

def demonstrate_free_guard_zone():
    """演示自由防守区规则"""
    print("\n" + "=" * 50)
    print("演示自由防守区规则：第5壶前不能将对方冰壶从自由防守区移出")
    print("=" * 50)
    
    rules_manager = CurlingRulesManager()
    
    # 场景1：第3壶违例情况
    print("\n场景1：第3壶违例 - 将对方冰壶从自由防守区移出")
    shot_number = 3
    moved_stones = [
        {
            'stone_id': 'team2_stone_1',
            'team': 'team2',
            'original_position': [2.5, 7.5],  # 在自由防守区内（T线和前掷线之间）
            'final_position': None  # 被移出场外
        }
    ]
    team_throwing = 'team1'
    
    has_violation, violations = rules_manager.check_free_guard_zone_violation(
        shot_number, moved_stones, team_throwing
    )
    
    print(f"是否违例: {has_violation}")
    if has_violation:
        print("违例详情:")
        for v in violations:
            print(f"  - {v['stone_id']} 从 {v['original_position']} 被移出场外")
        
        penalty_actions = rules_manager.apply_free_guard_zone_penalty(violations, 'thrown_stone_1')
        print("处罚措施:")
        for action in penalty_actions:
            print(f"  - {action['action']}: {action.get('stone', action.get('position', 'N/A'))}")
    
    # 场景2：第6壶正常情况
    print("\n场景2：第6壶 - 不适用自由防守区规则")
    shot_number = 6
    has_violation, violations = rules_manager.check_free_guard_zone_violation(
        shot_number, moved_stones, team_throwing
    )
    print(f"是否违例: {has_violation}")
    
    # 场景3：在自由防守区内移动但未移出
    print("\n场景3：在自由防守区内移动但未移出 - 不违例")
    shot_number = 2
    moved_stones_no_violation = [
        {
            'stone_id': 'team2_stone_1',
            'team': 'team2',
            'original_position': [2.5, 7.5],  # 在自由防守区内
            'final_position': [2.6, 7.6]     # 仍在自由防守区内
        }
    ]
    
    has_violation, violations = rules_manager.check_free_guard_zone_violation(
        shot_number, moved_stones_no_violation, team_throwing
    )
    print(f"是否违例: {has_violation}")

def demonstrate_complete_game():
    """演示完整比赛流程"""
    print("\n" + "=" * 50)
    print("演示完整比赛流程")
    print("=" * 50)
    
    rules_manager = CurlingRulesManager()
    rules_manager.reset_game('team1')
    
    # 模拟3局比赛
    game_results = [
        (2, 0, "Team1获胜2分"),
        (0, 1, "Team2获胜1分"),
        (1, 1, "平局")
    ]
    
    for end_num, (team1_score, team2_score, description) in enumerate(game_results, 1):
        print(f"\n第{end_num}局开始，先手: {rules_manager.first_hand_team}")
        print(f"第{end_num}局结果: {description}")
        
        # 模拟冰壶位置
        team1_stones = [[2.4 + i*0.1, 4.9 + i*0.1] for i in range(team1_score)] + [None] * (8 - team1_score)
        team2_stones = [[2.5 + i*0.1, 5.0 + i*0.1] for i in range(team2_score)] + [None] * (8 - team2_score)
        
        rules_manager.update_game_state(team1_stones, team2_stones)
        
        # 显示当前状态
        end_scores = rules_manager.end_scores[-1]
        print(f"累计得分: Team1={end_scores['cumulative_team1']}, Team2={end_scores['cumulative_team2']}")
    
    # 显示比赛总结
    summary = rules_manager.get_game_summary()
    print(f"\n比赛总结:")
    print(f"最终得分: Team1={summary['final_scores']['team1']}, Team2={summary['final_scores']['team2']}")
    print(f"获胜者: {summary['winner']}")
    print(f"总局数: {summary['total_ends']}")

if __name__ == "__main__":
    print("冰壶比赛规则演示程序")
    print("实现用户描述的冰壶比赛规则：")
    print("1. 得分规则：以场地上冰壶距离大本营圆心远近决定胜负")
    print("2. 先后手规则：得分队伍下一局为先手，平局则交换先后手")
    print("3. 自由防守区规则：第5壶前不能将对方冰壶从自由防守区移出")
    
    demonstrate_scoring_rules()
    demonstrate_first_hand_rules()
    demonstrate_free_guard_zone()
    demonstrate_complete_game()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)