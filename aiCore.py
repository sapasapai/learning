from langchain_community.vectorstores import HanaDB
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.service import OrchestrationService
import os

#fill with relavent AI Core credentials

os.environ["AICORE_AUTH_URL"] = ""
os.environ["AICORE_CLIENT_ID"] = ""
os.environ["AICORE_CLIENT_SECRET"] = ""
os.environ["AICORE_RESOURCE_GROUP"] = "UK001-SCB"
os.environ["AICORE_BASE_URL"] = ""


def llm_call(system_prompt, user_content):

    llm = LLM(
        name="mistralai--mistral-large-instruct",
        # parameters={
        #     'temperature': 0.0,
        # }
    )
    template = Template(
                messages=[
                    SystemMessage('''{{ ?system_prompt }}''' ),
                    UserMessage("""{{ ?user_content }}"""),
                ]
            )


    config = OrchestrationConfig(
        template=template,
        llm=llm,

    )
    orchestration_service = OrchestrationService(
        api_url="https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d086517115b8365b",
        config=config
    )

    response = orchestration_service.run(
        template_values=[
            TemplateValue(
                name="system_prompt",
                value= system_prompt
            ),
            TemplateValue(
                name="user_content",
                value= user_content
            )

        ]
    )

    #print(response.orchestration_result.choices[0].message.content)
    return(response.orchestration_result.choices[0].message.content)


