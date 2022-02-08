# msagent: 

A lightweight training framework of distributed reinforcement learning based on microservices

## Background: The necessity of league training

At present, the game decision-making technology mainly focuses on the two person zero sum game, especially some complex imperfect information games. The existing technical challenges include  
the explosive growth of the number of information sets in the mode of exponential function, huge observation and action space, sparse reward, long-term decision-making, the coexistence of 
transitive strategy sets and non transitive strategy sets, and so on. Among them, AI technology represented by alphastar, openai five reveals that the training process of AI with high 
generalization and robustness is inseparable from the construction of diversity strategy set.


Deepmind has a vivid "spinning tops model" description. The width of the spinning  represents the diversity of strategies, and the height of the 
spinning represents the skill level. In order to achieve the final Nash equilibrium, we must rely on enough strategy sets.



<div style="align: center">
<img src=pic/spinning_tops.png>
</div>


## Feature:

The efficiency and ease of use of the framework can be summarized as follows:

1. Support the alliance training mode: the opponent selection mechanism based on PFSP and the memory database are used as the model pool to realize higher speed response to requests and lower latency reading and writing.

2. Modular design concept and microservice paradigm: completely split the actor, learner and environment of reinforcement learning, and each component can be customized and encapsulated into corresponding services, so as to realize service registration, service expansion and service discovery with microservice architecture.

3. Efficient gradient update method: compared with PS computing model, it is prone to local hot spots leading to accelerated deterioration. Ring-allreduce model is adopted to make full use of the bandwidth between nodes in the cluster.

4. High throughput and high concurrency: it supports synchronous tasks and asynchronous tasks at the same time. It adopts the concept of message queue and task queue and combines the message routing mechanism to efficiently complete task scheduling and execution.

The framework simplifies the difficulty of distributed reinforcement learning training. When using the framework, users only need to write service script and environment script based on the provided interface, specify script path, startup quantity and other relevant parameters respectively, and then they can start the training with one click, which improves the ease of use.

## principle：

**Flexibility**: modular decoupling design and dynamic scalability.

**High efficiency**: low communication overhead, support large-scale parallel sampling and training.

**Ease of use**: Code lightweight and module customizability

## Usage:

Before you use it, you should ensure that you have installed consul for service registration and service discovery, and redis for storing model checkpoints during training.

Two possible environments are fighting game environment and football environment,the 

installation && configuration of the corresponding environment are as follows, 

which can be handled flexibly according to your own needs:

FightingICE: https://github.com/TeamFightingICE/FightingICE

GFR: https://github.com/google-research/football

### **Tips**:

When you run the FightingICE environment on the server, there may be errors like' py4j. protocol. Py4JNetWorkError: An error occurred while trying to the Java server(ip:port)'

In fact, this is because the server lacks GUI. If you are not convenient to install GUI on the server, there is another feasible way to choose.

First, you need to modify the source code of FightingIce to make the created graphical interface invisible,

Secondly, you need to use Xvfb to set DISPLAY variable, Xvfb is an X server that can run on machines with no display hardware and no physical input devices It emulates a dumb framebuffer using virtual memory.

Finally, you need to add the option --disable-window to avoid to create a graphical interface continuously during the running of the environment.

## Provide the distributed communication interface 

```
class Worker(object):
  def __init__(self, *args, **kwargs):
    super().__init__()
  
  @register(response=True)
  '''
  the register decorator can register the method to a available service,

  params:

  response = True: define the method to a synchronous task, which will block until get the result,
  reponse = False: define the method to a asynchronous task
  batch: When it is greater than 1, the requests will wait for batch clients synchronizelly.

  '''
  def func(self, *args, **kwargs):
     .......

from msagent.services.solver import Solver
worker = Worker()
solver = Solver()
ans = sovler.retrive(worker, 'func', *(args,), **kwargs)
```
This programming paradigm is particularly practical in reinforcement learning. Usually, for large-scale distributed reinforcement learning, an actor
needs to correspond to multiple environments, which is similar to the idea of SEEDRL, that is, the environment sends observation information to the actor, 
the actor performs the forward operation, sends the action information back to the environment. This process will continue until the end of the environment.
Manually maintaining the mapping the information is not only cumbersome and error prone, but also not easy to expand. 
In msagent, it is solved by registering functions and using Solver, the user can get the answer directly and don't need to care about which actor performs the forward operation,
Another problem is how the actor gets the latest parameters. msagent adopts the publish-subscribe mode, after training, 
the learner will automatically publish the latest parameters, which can only be obtained by actors who subscribe to the specified topic,
It is convenient to support the synchronous algorithm and asynchronous algorithm in reinforcement learning (similar to A3C). 
The current version needs to be revised in the source code, but in subsequent versions, we will provide a way that can be configured freely.


## Provide the abstraction of model and algorithms

The pyramid structure of basic operator - > network model - > interactive strategy - > training algorithm is provided, 
so that users can quickly modify and design the required network and algorithm according to their own needs, 
which reduce the difficulty of users. You can refer to the msagent/policy directory.

## Authors:

The project leader is Junge Zhang , and the main contributors are Bailin Wang and Kaijie Yang, Qingling Li. Kaiqi Huang co-supervises this project as well. In recent years, this team has been devoting to reinforcement learning, multi-agent system, decision-making intelligence