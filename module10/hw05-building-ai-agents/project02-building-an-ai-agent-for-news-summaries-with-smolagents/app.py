import yaml
import os
from smolagents import GradioUI, CodeAgent, TransformersModel

# Get current directory path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from tools.fetch_latest_news_titles_and_urls import SimpleTool as FetchLatestNewsTitlesAndUrls
from tools.summarize_news import SimpleTool as SummarizeNews
from tools.extract_news_article_content import SimpleTool as ExtractNewsArticleContent
from tools.classify_topic import SimpleTool as ClassifyTopic
from tools.final_answer import FinalAnswerTool as FinalAnswer



model = TransformersModel(
max_new_tokens=3000,
model_id='Qwen/Qwen2.5-Coder-7B-Instruct',
)

fetch_latest_news_titles_and_urls = FetchLatestNewsTitlesAndUrls()
summarize_news = SummarizeNews()
extract_news_article_content = ExtractNewsArticleContent()
classify_topic = ClassifyTopic()
final_answer = FinalAnswer()


with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent_news_agent = CodeAgent(
    model=model,
    tools=[fetch_latest_news_titles_and_urls, summarize_news, extract_news_article_content, classify_topic],
    managed_agents=[],
    max_steps=20,
    verbosity_level=2,
    grammar=None,
    planning_interval=None,
    name='news_agent',
    description='This agent is a smart news aggregator that fetches, summarizes, and classifies real-time news updates.',
    executor_type='local',
    executor_kwargs={},
    max_print_outputs_length=None,
    prompt_templates=prompt_templates
)
if __name__ == "__main__":
    GradioUI(agent_news_agent).launch()
