class CurlingRulesManager:
    """
    冰壶比赛规则管理器
    实现用户描述的冰壶比赛规则：
    1. 得分规则：位于大本营中且距离圆心更近的冰壶得分
    2. 先后手规则：得分队伍下一局为先手，平局则交换先后手
    3. 自由防守区规则：第5壶前不能将对方冰壶从自由防守区移出
    """
    
    def __init__(self):
        # 大本营圆心坐标 (x, y)
        self.house_center = [2.375, 4.88]
        # 大本营半径
        self.house_radius = 1.83
        # 自由防守区边界（T线到前掷线之间，除大本营外的区域）
        self.free_guard_zone = {
            'min_y': 6.71,  # T线到前掷线之间的区域开始
            'max_y': 9.75,  # 前掷线位置
            'min_x': 0.5,   # 左边界
            'max_x': 4.25   # 右边界
        }
        
        # 比赛状态
        self.current_end = 1
        self.max_ends = 16
        self.scores = {'team1': 0, 'team2': 0}
        self.end_scores = []  # 每局得分记录
        self.first_hand_team = None  # 当前局先手队伍
        self.last_winner = None  # 上一局获胜队伍
        
    def calculate_distance_to_center(self, stone_pos):
        """计算冰壶到大本营圆心的距离"""
        return ((stone_pos[0] - self.house_center[0])**2 + 
                (stone_pos[1] - self.house_center[1])**2)**0.5
    
    def is_stone_in_house(self, stone_pos):
        """判断冰壶是否在大本营内"""
        distance = self.calculate_distance_to_center(stone_pos)
        return distance <= self.house_radius
    
    def is_stone_in_free_guard_zone(self, stone_pos):
        """判断冰壶是否在自由防守区内"""
        x, y = stone_pos
        
        # 检查是否在自由防守区的Y坐标范围内
        if not (self.free_guard_zone['min_y'] <= y <= self.free_guard_zone['max_y']):
            return False
            
        # 检查是否在自由防守区的X坐标范围内
        if not (self.free_guard_zone['min_x'] <= x <= self.free_guard_zone['max_x']):
            return False
            
        # 检查是否在大本营内（大本营不属于自由防守区）
        if self.is_stone_in_house(stone_pos):
            return False
            
        return True
    
    def calculate_end_score(self, team1_stones, team2_stones):
        """
        计算一局的得分
        每只位于大本营中、位置较另外一队所有壶都更接近圆心的壶可记为得分壶
        """
        # 筛选出在大本营内的冰壶
        team1_in_house = []
        team2_in_house = []
        
        for stone in team1_stones:
            if stone and self.is_stone_in_house(stone):
                distance = self.calculate_distance_to_center(stone)
                team1_in_house.append((stone, distance))
                
        for stone in team2_stones:
            if stone and self.is_stone_in_house(stone):
                distance = self.calculate_distance_to_center(stone)
                team2_in_house.append((stone, distance))
        
        # 如果两队都没有冰壶在大本营内，平局
        if not team1_in_house and not team2_in_house:
            return 0, 0
        
        # 如果只有一队有冰壶在大本营内
        if team1_in_house and not team2_in_house:
            return len(team1_in_house), 0
        if team2_in_house and not team1_in_house:
            return 0, len(team2_in_house)
        
        # 两队都有冰壶在大本营内，计算得分
        team1_in_house.sort(key=lambda x: x[1])  # 按距离排序
        team2_in_house.sort(key=lambda x: x[1])
        
        # 找到距离圆心最近的冰壶
        closest_distance = min(team1_in_house[0][1], team2_in_house[0][1])
        
        team1_score = 0
        team2_score = 0
        
        # 计算team1得分
        if team1_in_house[0][1] <= team2_in_house[0][1]:
            for _, distance in team1_in_house:
                if distance < team2_in_house[0][1]:
                    team1_score += 1
                else:
                    break
        
        # 计算team2得分
        if team2_in_house[0][1] <= team1_in_house[0][1]:
            for _, distance in team2_in_house:
                if distance < team1_in_house[0][1]:
                    team2_score += 1
                else:
                    break
        
        return team1_score, team2_score
    
    def determine_next_first_hand(self, team1_score, team2_score):
        """
        确定下一局的先手队伍
        得分的队伍在下一局中是先手，如平局则双方交换先后手
        """
        if team1_score > team2_score:
            # Team1获胜，Team1下一局先手
            self.first_hand_team = 'team1'
            self.last_winner = 'team1'
        elif team2_score > team1_score:
            # Team2获胜，Team2下一局先手
            self.first_hand_team = 'team2'
            self.last_winner = 'team2'
        else:
            # 平局，交换先后手
            if self.first_hand_team == 'team1':
                self.first_hand_team = 'team2'
            else:
                self.first_hand_team = 'team1'
            self.last_winner = None
    
    def check_free_guard_zone_violation(self, shot_number, moved_stones, team_throwing):
        """
        检查自由防守区违例
        在第5壶（先手方第3壶）之前，不能将对方冰壶从自由防守区移出
        """
        # 只在前4壶检查（先手方前2壶，后手方前2壶）
        if shot_number >= 5:
            return False, []
        
        violations = []
        
        for stone_info in moved_stones:
            original_pos = stone_info['original_position']
            final_pos = stone_info['final_position']
            stone_team = stone_info['team']
            
            # 只检查对方的冰壶
            if stone_team == team_throwing:
                continue
                
            # 检查是否从自由防守区移出
            was_in_fgz = self.is_stone_in_free_guard_zone(original_pos)
            is_still_in_play = final_pos is not None  # None表示出界
            
            if was_in_fgz and not is_still_in_play:
                violations.append(stone_info)
        
        return len(violations) > 0, violations
    
    def apply_free_guard_zone_penalty(self, violations, thrown_stone):
        """
        应用自由防守区违例处罚
        该投出的壶拿开，其余被触及的壶将放回违例发生前的位置
        """
        penalty_actions = []
        
        # 移除投出的冰壶
        penalty_actions.append({
            'action': 'remove_stone',
            'stone': thrown_stone,
            'reason': 'free_guard_zone_violation'
        })
        
        # 恢复被违规移动的冰壶
        for violation in violations:
            penalty_actions.append({
                'action': 'restore_stone',
                'stone': violation['stone_id'],
                'position': violation['original_position'],
                'reason': 'free_guard_zone_violation'
            })
        
        return penalty_actions
    
    def update_game_state(self, team1_stones, team2_stones):
        """更新比赛状态"""
        # 计算当前局得分
        team1_score, team2_score = self.calculate_end_score(team1_stones, team2_stones)
        
        # 更新总分
        self.scores['team1'] += team1_score
        self.scores['team2'] += team2_score
        
        # 记录本局得分
        self.end_scores.append({
            'end': self.current_end,
            'team1_score': team1_score,
            'team2_score': team2_score,
            'cumulative_team1': self.scores['team1'],
            'cumulative_team2': self.scores['team2']
        })
        
        # 确定下一局先手
        if self.current_end < self.max_ends:
            self.determine_next_first_hand(team1_score, team2_score)
        
        # 准备下一局
        self.current_end += 1
    
    def is_game_over(self):
        """判断比赛是否结束"""
        return self.current_end > self.max_ends
    
    def get_winner(self):
        """获取比赛获胜者"""
        if not self.is_game_over():
            return None
        
        if self.scores['team1'] > self.scores['team2']:
            return 'team1'
        elif self.scores['team2'] > self.scores['team1']:
            return 'team2'
        else:
            return 'tie'
    
    def get_game_summary(self):
        """获取比赛总结"""
        return {
            'final_scores': self.scores,
            'winner': self.get_winner(),
            'total_ends': self.current_end - 1,
            'end_by_end_scores': self.end_scores,
            'is_completed': self.is_game_over()
        }
    
    def reset_game(self, first_hand_team='random'):
        """重置比赛状态"""
        self.current_end = 1
        self.scores = {'team1': 0, 'team2': 0}
        self.end_scores = []
        self.last_winner = None
        
        if first_hand_team == 'random':
            import random
            self.first_hand_team = random.choice(['team1', 'team2'])
        else:
            self.first_hand_team = first_hand_team
        
        print(f"新比赛开始，{self.first_hand_team} 先手")