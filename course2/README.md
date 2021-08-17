# 习题课第二天

## 任务：环境悬崖漫步 - 算法Q-learning & SARSA - 提交到Jidi平台，成绩优于随机10%

提交链接：http://www.jidiai.cn/cliffwalking


---
## Env 👉请看 [cliffwalking.py](env/cliffwalking.py)
## Q-learning 👉请看 [tabularq.py](examples/algo/tabularq/tabularq.py)
## Sarsa 👉请看 [sarsa.py](examples/algo/sarsa/sarsa.py)
## Homework 👉请看 [submission.py](examples/algo/homework/submission.py)

---
# How to train your rl_agent:

Have a go~
>python main.py --scenario cliffwalking --algo sarsa --reload_config

>python main.py --scenario cliffwalking --algo tabularq --reload_config

说明：
1. 算法需要在本地训练，及第平台提供了经典算法实现、训练框架和提交样例。
2. 在config文件夹里，已经保存了算法库对接多个环境和多个算法的训练参数。支持一键复现，只需要加 --reload_config这个参数（So cool...
3. 训练开始后，会生成models文件夹，在models/config_training里面保存了训练过程中的参数。可以试着不加reload_config，就在👈里调参，主run会自动上传这里的参数：例如python main.py --scenario cliffwalking --algo sarsa

---
# How to test submission

Complete submission.py, and then
>python run_log.py 

If no errors, your submission is ready to go~

---
# Bonus
gridworld和cliffwalking都是网格环境，智能体tabularq依然是“冒险家“，sarsa还是“保险主义”。运行试试吧^0^
