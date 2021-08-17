# 习题课第三天

## 任务：车杆cartpole - 算法DQN - 提交到Jidi平台，成绩优于随机10%

提交链接：http://www.jidiai.cn/cartpole


---
## DQN 👉请看 examples/algo/dqn/dqn.py

---
# How to train your rl_agent:

have a go~
>python main.py --scenario classic_CartPole-v0 --algo dqn --reload_config 

We also provide 2 DQN variants - DDQN & Dueling DQN.
>python main.py --scenario classic_CartPole-v0 --algo ddqn --reload_config 

>python main.py --scenario classic_CartPole-v0 --algo duelingq --reload_config 

说明：
1. 算法需要在本地训练，及第平台提供了经典算法实现、训练框架和提交样例。
2. 在config文件夹里，已经保存了算法库对接多个环境和多个算法的训练参数。支持一键复现，只需要加 --reload_config这个参数（So cool...
3. 训练开始后，会生成models文件夹，在models/config_training里面保存了训练过程中的参数。可以试着不加reload_config，就在👈里调参，主run会自动上传这里的参数：例如python main.py --scenario cliffwalking --algo sarsa

---
# Bonus

