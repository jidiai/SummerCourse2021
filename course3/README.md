# 习题课第三天

## 任务：车杆cartpole - 算法DQN - 提交到Jidi平台，成绩优于随机10%

提交链接：http://www.jidiai.cn/cartpole


---
## DQN 👉请看 [dqn.py](examples/algo/dqn/dqn.py)
## Homework 👉请看 [submission.py](examples/algo/homework/submission.py)
---
# How to train your rl_agent:

Have a go~
>python main.py --scenario classic_CartPole-v0 --algo dqn --reload_config 

We also provide 2 DQN variants - DDQN & Dueling DQN.
>python main.py --scenario classic_CartPole-v0 --algo ddqn --reload_config 

>python main.py --scenario classic_CartPole-v0 --algo duelingq --reload_config 

Tips：
1. 调参小技巧：加上reload_config读的是config下调好的参数；不加是读的model/config_training下的参数；每次跑参数还会在models/Cartpole/run/文件加下保存哟~
2. 通过tensorboard看训练曲线：tensorboard --logdir=models 

---
# Bonus
尝试训练DQN在不同环境吧，eg.classic_MountainCar-v0，一起玩转及第金榜~

完整环境和算法库链接：https://github.com/jidiai/ai_lib

