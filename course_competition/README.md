# RLChina2021 SummerCourse 大作业

## 任务：贪吃蛇 snakes_1v1 - 算法自定义 - 提交到Jidi平台，成绩优于随机
提交链接：[RLChina2021暑期习题课擂台](http://www.jidiai.cn/compete_detail?compete=7#competitionRule)


### Dependency
You need to create competition environment.
>conda create -n snake1v1 python=3.6

>conda activate snake1v1

>pip install -r requirements.txt

---

## Baseline 👉请看[random_agent.py](agent/random/random_agent.py)
## Homework 👉请看[submission.py](agent/homework/submission.py)

---
### How to train rl-agent

>python rl_trainer/main.py

You can edit different parameters, for example

>python rl_trainer/main.py --lr_a 0.001 --seed_nn 2


### How to test submission 

When your submission is ready, you can locally test your submission. At Jidi platform, we evaluate your submission as same as **run_log.py**

Once you run this file, you can locally check battle logs in the folder named "logs".

Have fun~




