# federated learning

https://github.com/adap/flower

# ai agent

https://ai-guidebook-nexus.lovable.app/frameworks

## taiwan course.

### lesson 1

https://colab.research.google.com/drive/1EjiX46muxSMy0avtHPXiulVUhmu37Kyi?usp=sharing#scrollTo=EIaygQUTLxRi

### lesson 2

https://colab.research.google.com/drive/1t347cQEyMikpHUHV_ap83A-mq9PMSl3O?usp=sharing

prompt engineering VS context engineering
give a better input to model. avoid put too much context into prompt.

now LLM is good enough, it can understand the context as human.
so we don't need to give a better input to model.

context engineering: automatically manage the prompt input.

context could be very long, like 250K token.

context engineering:

1. save context to file
2. select context (static select, dynamic select)
3. compress context
4. isolate context

system prompt is very complicated now. some items as follows:

1. basic product info
2. usage limitation
3. ask user give feedback
4. security
5. response style (for example, don't use good question)
6.

long term memory: record the conversation history.

provide LLM extra information, for example from Internet.

RAG

Agentic workflow: fix problem according to SOP, fixed steps

#### AI agent

AI agent: AI agent can think step by step. and include some loop. repeat thinking, process and action.

AI agent have three parts, do things step by step, and repeat then until Goal is reached.

- Goal
- Observations (get the information from environment, get the failed test case and error message)
- Actions (take action to environment, like update the test file then run a test case)

RL: reinforcement learning, set the reward to guide the AI agent to reach the goal.
for example we play the go game, the reward is the probability of winning.

#### Multi agent

single agent VS multi agent

https://stream-bench.github.io/
StreamBench to test the LLM.
