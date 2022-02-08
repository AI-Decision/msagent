class Action(object):
    def __init__(self):

        self._actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
        self._action_strs = self._actions.split(" ")

    def get_action_strs(self):
        return self._action_strs

    def check_vaild_action(self, act_num, obs):
        eT5 = obs[2]
        eT30 = obs[3]
        eT50 = obs[4]
        eT150 = obs[5]
        state = obs[68:72]
        
        res_act = act_num
        if state[2] == 1:       # on AIR
            if act_num > 17:
                res_act = 0     # action 'AIR'
            else:
                # Energy Check
                if eT50 < 1:
                    if act_num in [3, 4, 5, 6, 9, 10]:
                        res_act = 0
        else:                   # on GROUND
            if act_num <= 17:
                res_act = 35    # NEUTRAL
            else:
                # Energy Check
                if eT150 < 1:
                    if act_num in [40, 41, 42, 43, 44, 45, 46, 52, 53]:
                        res_act = 35


        return res_act

    
    def check_vaild_action_zen(self, act_num, obs):
        # 4.50
        eT5 = obs[2]
        eT30 = obs[3]
        eT50 = obs[4]
        eT150 = obs[5]
        state = obs[68:72]
        
        res_act = act_num
        if state[2] == 1:       # on AIR
            if act_num > 17:
                res_act = 0     # action 'AIR'
            else:
                # Energy Check
                if eT5 < 1:
                    if act_num in [5]:
                        res_act = 0
                if eT30 < 1:
                    if act_num in [6, 9, 3]:
                        res_act = 0
                if eT50 < 1:
                    if act_num in [4, 10]:
                        res_act = 0
        else:                   # on GROUND
            if act_num <= 17:
                res_act = 35    # NEUTRAL
            else:
                # Energy Check
                if eT5 < 1:
                    if act_num in [42 ,52]:
                        res_act = 35
                if eT30 < 1:
                    if act_num in [43, 53]:
                        res_act = 35
                if eT50 < 1:
                    if act_num in [41, 46]:
                        res_act = 35
                if eT150 < 1:
                    if act_num in [44]:
                        res_act = 35

        return res_act
