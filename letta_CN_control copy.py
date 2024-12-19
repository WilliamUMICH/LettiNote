import os, json
from datetime import datetime
from letta import create_client, EmbeddingConfig, LLMConfig, ChatMemory
import sys
from letta import AgentState

client = create_client()

client.set_default_embedding_config(
    EmbeddingConfig.default_config(model_name="letta")
)
client.set_default_llm_config(LLMConfig.default_config(model_name="letta"))

def create_CN(self, ) -> str:
    """
    When users want to generate clinical notes, call this function.

    """
    import os, json, logging, sys
    sys.path.append(os.path.abspath("/Users/williamzheng/Documents/UmichFolder/2025 Fall Semester /Research 499/LettiNote"))
    logging.basicConfig(level=logging.INFO,  format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info('HEERREEEE IIII AMMM' + os.getcwd())

    import utils
    import utils.chat 
    import utils.data
    from utils.infer import infer_openai
    # from utils.infer import *

    # from utils import *
    print("WORKING PRINT: ", os.getcwd())
    
    inputPath = "data/summaries/augmented_notes_small.jsonl" 
    outputPath = 'data/inference/gpt3-direct.jsonl'
    modelName = "gpt3"
    mode = "direct-gpt" # :param mode: str, mode of inference (direct-gpt, generator-gpt)
    max_tokens = 20000

    # Generate clinical notes from conversations using an OpenAI model.

    try:
        print('TESTPRINT')
    except:
        print('FALSE PRINT')

    try:
        logging.info('NOTE 12: running inference')
        infer_openai(
            input_path=inputPath,
            output_path=outputPath,
            model_name=modelName,
            mode=mode,
            max_tokens=max_tokens
        )
    except Exception as e:
        logging.debug('ERROR MESSAGE FULL: ', e)



    return None

def give_workingDirectory(self, ) -> str:
    """
    Return the current working directory

    Return
        directory (str): the path of working directory
    """
    
    import os
    print(os.getcwd())
    return os.getcwd()

generate_CN = client.create_tool(create_CN)
getCWD = client.create_tool(give_workingDirectory)


agent_name = "LettiNote"

try:
    agent_state = client.create_agent(
        name=agent_name,
        tools=[generate_CN.name]
    )
except ValueError:
    agent_id = client.get_agent_id(agent_name)
    client.delete_agent(agent_id)
    agent_state = client.create_agent(
        agent_name,
        tools=[generate_CN.name]
    )