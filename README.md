# iKLLM_Email_Agent
pipeline with local iKraph, LLM, email_agent


### Main Goal
When user ask a question to iKraph with email sending request, the Router divide it into pure medical question (for iKraph) and email intent (for Email Agent). The LLM will make response to medical question and also email the content to target address.
e.g. "What is the application of Panadol? Please send the answer to eric85021811@gmail.com."

<p align="center">
  <img src="./images/email_agent_pipeline.png" alt="email_agent_pipeline" width="700" height="350"/>
</p>

We have our local iKraph (Neo4j-based) database
<p align="center">
  <img src="./images/local_ikraph_UI.png" alt="local_ikraph_UI" width="700" height="350"/>
</p>

Workflow for handling medical question
<p align="center">
  <img src="./images/IK_workflow.png" alt="IK_workflow" width="700" height="350"/>
</p>
---
