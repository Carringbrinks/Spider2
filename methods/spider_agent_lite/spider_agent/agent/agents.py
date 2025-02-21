import logging
import logging
import os
import platform
import re
from typing import Dict, List

from methods.spider_agent_lite.spider_agent.agent.action import Action, Bash, Terminate, CreateFile, EditFile, \
    LOCAL_DB_SQL, BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, SelectTable
from methods.spider_agent_lite.spider_agent.agent.models import call_llm
from methods.spider_agent_lite.spider_agent.agent.prompts import LOCAL_SYSTEM, DBT_SYSTEM, REFERENCE_PLAN_SYSTEM, \
    BIGQUERY_SYSTEM_ZH, SNOWFLAKE_SYSTEM_ZH, DDL_TEMPLATE
from methods.spider_agent_lite.spider_agent.envs.spider_agent import Spider_Agent_Env

logger = logging.getLogger("spider_agent")

def get_ddl_file_path(work_dir):
    sub_directory = os.listdir(work_dir)
    if not sub_directory:
        return {}, {}

    md_content = {}
    csv_content = {}

    for unknow_path in sub_directory:
        full_path = os.path.join(work_dir, unknow_path)

        if os.path.isfile(full_path):
            if unknow_path.endswith(".md"):
                with open(full_path, "r", encoding="utf-8") as file_md:
                    md_content[full_path] = file_md.read()

            elif unknow_path.endswith(".csv"):
                with open(full_path, "r", encoding="utf-8") as file_csv:
                    csv_content[full_path] = file_csv.read()
        elif os.path.isdir(full_path):
            sub_md, sub_csv = get_ddl_file_path(full_path)
            md_content.update(sub_md)
            csv_content.update(sub_csv)

    return md_content, csv_content


class PromptAgent:
    def __init__(
        self,
        model="gpt-4",
        max_tokens=1500,
        top_p=0.9,
        temperature=0.5,
        max_memory_length=10,
        max_steps=15,
        use_plan=False
    ):
        
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.max_memory_length = max_memory_length
        self.max_steps = max_steps
        
        self.thoughts = []
        self.responses = []
        self.actions = []
        self.observations = []
        self.system_message = ""
        self.history_messages = []
        self.env = None
        self.codes = []
        self.work_dir = "/workspace"
        self.use_plan = use_plan

    # TODO: ADD ACTION SPACE
    def set_env_and_task(self, env: Spider_Agent_Env):
        self.env = env
        self.thoughts = []
        self.responses = []
        self.actions = []
        self.observations = []
        self.codes = []
        self.history_messages = []
        self.instruction = self.env.task_config['question']

        database_dir = self.env.task_config["config"][0]["parameters"]["dirs"][0]
        # if self.env.task_config['type'] == 'Bigquery':
        #     self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, BIGQUERY_EXEC_SQL, CreateFile, EditFile]
        #     action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        #     self.system_message = BIGQUERY_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        # elif self.env.task_config['type'] == 'Snowflake':
        #     self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, SNOWFLAKE_EXEC_SQL, CreateFile, EditFile]
        #     action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        #     self.system_message = SNOWFLAKE_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        if self.env.task_config['type'] == 'Bigquery':
            ddl_info = []
            database_md, ddl_csv = get_ddl_file_path(database_dir)
            
            for key, value in ddl_csv.items():
                if platform.system() == "Windows":
                    ddl_info.append(DDL_TEMPLATE.format(database=key.split("\\")[-2], ddl=value))
                else:
                    ddl_info.append(DDL_TEMPLATE.format(database=key.split("/")[-2], ddl=value))
            self._AVAILABLE_ACTION_CLASSES = [SelectTable, Terminate, BIGQUERY_EXEC_SQL, CreateFile, EditFile]
            action_space = "".join(
                [action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
            self.system_message = BIGQUERY_SYSTEM_ZH.format(work_dir=self.work_dir, action_space=action_space,
                                                            DescriptionMd = list(database_md.values())[0] if database_md else "", DDL="\n".join(ddl_info),
                                                         task=self.instruction, max_steps=self.max_steps)
        elif self.env.task_config['type'] == 'Snowflake':
            ddl_info = []
            database_md, ddl_csv = get_ddl_file_path(database_dir)
            if platform.system() == 'Windows':
                ddl_csv = {key.replace(database_dir + "\\", "").replace("\\DDL.csv", ""): value for key, value in
                           ddl_csv.items()}
            else:
                ddl_csv = {key.replace(database_dir + "/", "").replace("/DDL.csv", ""): value for key, value in
                           ddl_csv.items()}
            for key, value in ddl_csv.items():
                ddl_info.append( DDL_TEMPLATE.format(database=key, ddl=value))
            self._AVAILABLE_ACTION_CLASSES = [SelectTable, Terminate, SNOWFLAKE_EXEC_SQL, CreateFile, EditFile]
            action_space = "".join(
                [action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
            self.system_message = SNOWFLAKE_SYSTEM_ZH.format(work_dir=self.work_dir, action_space=action_space,
                                                             DescriptionMd=list(database_md.values())[0] if database_md else "", DDL="\n".join(ddl_info),
                                                          task=self.instruction, max_steps=self.max_steps)
        elif self.env.task_config['type'] == 'Local':
            self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL]
            action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
            self.system_message = LOCAL_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        elif self.env.task_config['type'] == 'DBT':
            self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL]
            action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
            self.system_message = DBT_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        
        if self.use_plan:
            self.system_message += REFERENCE_PLAN_SYSTEM.format(plan=self.reference_plan)
        

        
        self.history_messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": self.system_message 
                },
            ]
        })
        
    def predict(self, obs: Dict=None) -> List:
        """
        Predict the next action(s) based on the current observation.
        """    
        
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts) \
            , "The number of observations and actions should be the same."

        status = False
        while not status:
            messages = self.history_messages.copy()
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Observation: {}\n".format(str(obs))
                    }
                ]
            })  
            status, response = call_llm({
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            })
            response = response.strip()
            if not status:
                if response in ["context_length_exceeded","rate_limit_exceeded","max_tokens","unknown_error"]:
                    self.history_messages = [self.history_messages[0]] + self.history_messages[3:]
                else:
                    raise Exception(f"Failed to call LLM, response: {response}")
            

        try:
            action = self.parse_action(response)
            thought = re.search(r'Thought:(.*?)Action', response, flags=re.DOTALL)
            if thought:
                thought = thought.group(1).strip()
            else:
                thought = response
        except ValueError as e:
            print("Failed to parse action from response", e)
            action = None
        
        logger.info("Observation: %s", obs)
        logger.info("Response: %s", response)

        self._add_message(obs, thought, action)
        self.observations.append(obs)
        self.thoughts.append(thought)
        self.responses.append(response)
        self.actions.append(action)

        # if action is not None:
        #     self.codes.append(action.code)
        # else:
        #     self.codes.append(None)

        return response, action
        
    
    def _add_message(self, observations: str, thought: str, action: Action):
        self.history_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Observation: {}".format(observations)
                }
            ]
        })
        self.history_messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Thought: {}\n\nAction: {}".format(thought, str(action))
                }
            ]
        })
        if len(self.history_messages) > self.max_memory_length*2+1:
            self.history_messages = [self.history_messages[0]] + self.history_messages[-self.max_memory_length*2:]
    
    def parse_action(self, output: str) -> Action:
        """ Parse action from text """
        if output is None or len(output) == 0:
            pass
        action_string = ""
        patterns = [r'["\']?Action["\']?:? (.*?)Observation',r'["\']?Action["\']?:? (.*?)Thought', r'["\']?Action["\']?:? (.*?)$', r'^(.*?)Observation']

        for p in patterns:
            match = re.search(p, output, flags=re.DOTALL)
            if match:
                action_string = match.group(1).strip()
                break
        if action_string == "":
            action_string = output.strip()
        
        output_action = None
        for action_cls in self._AVAILABLE_ACTION_CLASSES:
            action = action_cls.parse_action_from_text(action_string)
            if action is not None:
                output_action = action
                break
        if output_action is None:
            action_string = action_string.replace("\_", "_").replace("'''","```")
            for action_cls in self._AVAILABLE_ACTION_CLASSES:
                action = action_cls.parse_action_from_text(action_string)
                if action is not None:
                    output_action = action
                    break
        
        return output_action
    

    
    def run(self):
        assert self.env is not None, "Environment is not set."
        result = ""
        done = False
        step_idx = 0
        obs = "Please start answering."
        retry_count = 0
        last_action = None
        repeat_action = False
        while not done and step_idx < self.max_steps:

            _, action = self.predict(
                obs
            )
            if action is None:
                logger.info("Failed to parse action from response, try again.")
                retry_count += 1
                if retry_count > 3:
                    logger.info("Failed to parse action from response, stop.")
                    break
                obs = "Failed to parse action from your response, make sure you provide a valid action."
            else:
                logger.info("Step %d: %s", step_idx + 1, action)
                if last_action is not None and last_action == action:
                    if repeat_action:
                        return False, "ERROR: Repeated action"
                    else:
                        obs = "The action is the same as the last one, you MUST provide a DIFFERENT SQL code or Python Code or different action. you MUST provide a DIFFERENT SQL code or Python Code or different action. you MUST provide a DIFFERENT SQL code or Python Code or different action."
                        repeat_action = True
                else:
                    obs, done = self.env.step(action)
                    last_action = action
                    repeat_action = False

            if done:
                if isinstance(action, Terminate):
                    result = action.output
                logger.info("The task is done.")
                break
            step_idx += 1

        return done, result

    def get_trajectory(self):
        trajectory = []
        for i in range(len(self.observations)):
            trajectory.append({
                "observation": self.observations[i],
                "thought": self.thoughts[i],
                "action": str(self.actions[i]),
                # "code": self.codes[i],
                "response": self.responses[i]
            })
        trajectory_log = {
            "Task": self.instruction,
            "system_message": self.system_message,
            "trajectory": trajectory
        }
        return trajectory_log


if __name__ == "__main__":
    agent = PromptAgent()
    response = """
BIGQUERY_EXEC_SQL(sql_query=\"\"\"
WITH purchase_users AS (
  SELECT DISTINCT user_pseudo_id
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE event_name = 'purchase' AND _TABLE_SUFFIX BETWEEN '20201201' AND '20201231'
),
pageviews AS (
  SELECT user_pseudo_id, COUNT(*) AS pageviews
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE event_name = 'page_view' AND _TABLE_SUFFIX BETWEEN '20201201' AND '20201231'
  GROUP BY user_pseudo_id
),
pageviews_by_user AS (
  SELECT 
    p.user_pseudo_id, 
    p.pageviews,
    CASE WHEN pu.user_pseudo_id IS NOT NULL THEN 'purchaser' ELSE 'non-purchaser' END AS user_type
  FROM pageviews p
  LEFT JOIN purchase_users pu ON p.user_pseudo_id = pu.user_pseudo_id
)
SELECT user_type, AVG(pageviews) AS avg_pageviews
FROM pageviews_by_user
GROUP BY user_type
\"\"\", is_save=True, save_path="avg_pageviews_dec_2020.csv")
"""

    response = """
BIGQUERY_EXEC_SQL(sql_query=\"\"\"
SELECT DISTINCT user_pseudo_id
FROM bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*
WHERE event_name = 'purchase' AND _TABLE_SUFFIX BETWEEN '20201201' AND '20201231'
\"\"\", is_save=False)
"""


    action = agent.parse_action(response)
    print(action)