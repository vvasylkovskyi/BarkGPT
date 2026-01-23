```sh
Deep Agent
   |
   |-- A2A send_message_streaming()
           |
           |-- Logs Agent returns TaskState.submitted + task_id
           |
           |-- Logs Agent continues work via Kafka
```

The flow that I have is the following:

1. The Logs Agent Executor receives a message, and writes the task to the topic=`a2a-agent-work-queue`
2. The Logs Agent Worker, consumes the topic=`a2a-agent-work-queue`, consumer group = `logs-agent-workers`, performs the task, and write to topic `a2a-agent-results`
3. The SRE Agent, consumes the topic `a2a-agent-results`, consumer group = `sre-agent-consumer-group`
