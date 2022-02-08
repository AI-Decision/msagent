import numpy as np


class CustomObservation(object):
    def __init__(self, b_simulate=True, b_stack=True, n_stack=5):

        self.b_simulate = b_simulate
        self.b_stack = b_stack
        self.n_stack = n_stack

        self.obs_dim = 165
        self.obs_stack = np.zeros((self.obs_dim*self.n_stack,))


    def set_player(self, _player):
        self.player = _player

    def set_simulator(self, _simulator):
        self.simulator = _simulator

    def reset_obs(self):
        self.obs_stack = np.zeros((self.obs_dim*self.n_stack,))

    def get_obs(self, frameData):

        if self.b_simulate:
            fd_proc = self.simulator.simulate(frameData, self.player, None, None, 13)
            obs = self._get_obs_data(frameData)
            obs_proc = self._get_obs_data(fd_proc)
            obs_res = np.concatenate((obs,obs_proc))

        else:
            fd_proc = frameData
    
            if self.b_stack:
                obs_res = self._get_obs_data_stack(fd_proc)
            else:
                obs_res = self._get_obs_data(fd_proc)

        return obs_res
    

    def _get_obs_data_stack(self, frameData):
        new_obs = self._get_obs_data(frameData)
        self.obs_stack[:-self.obs_dim] = self.obs_stack[self.obs_dim:]
        self.obs_stack[-self.obs_dim:] = new_obs
        return self.obs_stack

    def _get_obs_data(self, frameData):
        p1 = frameData.getCharacter(self.player)
        p2 = frameData.getCharacter(not self.player)

        if self.player:
            p1Projectiles = frameData.getProjectilesByP1()
            p2Projectiles = frameData.getProjectilesByP2()
        else:
            p2Projectiles = frameData.getProjectilesByP1()
            p1Projectiles = frameData.getProjectilesByP2()

        game_frame_num = frameData.getFramesNumber() / 3600

        observation = []
        dist = []
        for my in [p1, p2]:
            myHp = np.clip(my.getHp() / 400, 0, 1)
            myEnergy = my.getEnergy() / 300
            myEnergyT5 = my.getEnergy() / 5
            myEnergyT30 = my.getEnergy() / 30
            myEnergyT50 = my.getEnergy() / 50
            myEnergyT150 = my.getEnergy() / 150
            myX = ((my.getLeft() + my.getRight()) / 2) / 960
            myY = ((my.getBottom() + my.getTop()) / 2) / 640
            mySpeedX = my.getSpeedX() / 15
            mySpeedY = my.getSpeedY() / 28
            myAct = my.getAction().ordinal()
            myState = my.getState().ordinal()
            myRemainingFrame = my.getRemainingFrame() / 70
            myControllable = int(my.isControl())

            observation += [myHp, myEnergy, myEnergyT5, myEnergyT30,
                            myEnergyT50, myEnergyT150, myX, myY]
            if mySpeedX < 0:
                observation.append(0)
            else:
                observation.append(1)
            observation.append(abs(mySpeedX))
            if mySpeedY < 0:
                observation.append(0)
            else:
                observation.append(1)
            observation.append(abs(mySpeedY))
            myAct_onehot = [0]*56       # the number of actions
            myAct_onehot[myAct] = 1

            myState_onehot = [0]*4      # the number of states
            myState_onehot[myState] = 1
            observation += myAct_onehot
            observation += myState_onehot
            observation.append(myRemainingFrame)
            observation.append(myControllable)

            dist.append(myX)

        observation.append(game_frame_num)

        # distance
        dx = abs(dist[0]-dist[1])
        dx_onehot = [0]*3
        if dx <= 0.15:
            dx_onehot[0] = 1
        elif dx <= 0.3:
            dx_onehot[1] = 1
        else:
            dx_onehot[2] = 1
        observation.append(dx)
        observation += dx_onehot

        for myProjectiles in [p1Projectiles, p2Projectiles]:
            if len(myProjectiles) >= 2:
                myHitDamage = myProjectiles[0].getHitDamage() / 200.0
                myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                    0].getCurrentHitArea().getRight()) / 2) / 960.0
                myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                    0].getCurrentHitArea().getBottom()) / 2) / 640.0
                observation.append(myHitDamage)
                observation.append(myHitAreaNowX)
                observation.append(myHitAreaNowY)
                myHitDamage = myProjectiles[1].getHitDamage() / 200.0
                myHitAreaNowX = ((myProjectiles[1].getCurrentHitArea().getLeft() + myProjectiles[
                    1].getCurrentHitArea().getRight()) / 2) / 960.0
                myHitAreaNowY = ((myProjectiles[1].getCurrentHitArea().getTop() + myProjectiles[
                    1].getCurrentHitArea().getBottom()) / 2) / 640.0
                observation.append(myHitDamage)
                observation.append(myHitAreaNowX)
                observation.append(myHitAreaNowY)
            elif len(myProjectiles) == 1:
                myHitDamage = myProjectiles[0].getHitDamage() / 200.0
                myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                    0].getCurrentHitArea().getRight()) / 2) / 960.0
                myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                    0].getCurrentHitArea().getBottom()) / 2) / 640.0
                observation.append(myHitDamage)
                observation.append(myHitAreaNowX)
                observation.append(myHitAreaNowY)
                for t in range(3):
                    observation.append(0.0)
            else:
                for t in range(6):
                    observation.append(0.0)

        observation = np.array(observation, dtype=np.float32)
        observation = np.clip(observation, 0, 1)
        return observation
