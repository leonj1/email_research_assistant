#!/usr/bin/env python3
"""
Email script for generating and sending AI research summaries.

This script searches for AI-related content, summarizes it, and sends a daily email digest.
"""

from __future__ import print_function
import json
import os
import pathlib
import re
from typing import List, Dict, Any, Literal, Annotated

import requests
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
SEARCH_TERMS = [
    "Agentic AI",
    "OpenAI LinkedIn",
    "Perplexity LinkedIn",
    "Meta AI LinkedIn",
    "Anthropic LinkedIn"
]

required_environment_variables = [
    "SERPER_API_KEY",
    "SCRAPING_API_KEY",
    "SENDINGBLUE_API_KEY",
    "OPENAI_API_KEY",
    "DESTINATION_EMAIL"
]

def validate_environment_variables():
    """Validate environment variables."""
    logger.info("Validating environment variables")
    for var in required_environment_variables:
        if os.getenv(var) is None:
            logger.error(f"Environment variable {var} is not set")
            raise ValueError(f"Environment variable {var} is not set")
    logger.info("All environment variables validated successfully")


class ResultRelevance(BaseModel):
    """Model for storing relevance check results."""
    explanation: str
    id: str


class RelevanceCheckOutput(BaseModel):
    """Model for storing all relevant results."""
    relevant_results: List[ResultRelevance]


class State(TypedDict):
    """State management for the LangGraph workflow."""
    messages: Annotated[list, add_messages]
    summaries: List[dict]
    approved: bool
    created_summaries: Annotated[List[dict], Field(description="The summaries created by the summariser")]
    email_template: str


class SummariserOutput(BaseModel):
    """Output format for the summarizer."""
    email_summary: str = Field(description="The summary email of the content")
    message: str = Field(description="A message to the reviewer requesting feedback")


class ReviewerOutput(BaseModel):
    """Output format for the reviewer."""
    approved: bool = Field(description="Whether the summary is approved")
    message: str = Field(description="Feedback message from the reviewer")


def search_serper(search_query: str) -> List[Dict[str, Any]]:
    """Search Google using the Serper API."""
    logger.info(f"Searching Serper API for query: {search_query}")
    url = "https://google.serper.dev/search"
    
    payload = {
        "q": search_query,
        "gl": "gb",
        "num": 20,
        "type": "search"
    }
    logger.debug(f"Search payload: {payload}")

    headers = {
        'X-API-KEY': os.getenv("SERPER_API_KEY"),
        'Content-Type': 'application/json'
    }
    logger.debug("Making request to Serper API")
    
    response = requests.post(url, headers=headers, json=payload)
    results = response.json()
    
    if response.status_code != 200:
        logger.error(f"Serper API error: {results}")
        raise ValueError(f"Serper API error: {results}")
    
    organic_results = results.get('organic', [])
    logger.info(f"Found {len(organic_results)} results")
    logger.debug(f"Raw results: {results}")
    
    if not organic_results:
        logger.error(f"No organic results found in results {results}")
        raise ValueError(f"No organic results found in results {results}")
    
    # Add IDs to results
    processed_results = []
    for idx, result in enumerate(organic_results):
        processed_results.append({
            'id': str(idx),
            'title': result.get('title', ''),
            'link': result.get('link', ''),
            'snippet': result.get('snippet', ''),
            'search_term': search_query
        })
    
    return processed_results


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from file."""
    with open(f"prompts/{prompt_name}.md", "r") as file:
        return file.read()


def check_search_relevance(search_results: Dict[str, Any]) -> RelevanceCheckOutput:
    """Analyze search results and determine the most relevant ones."""
    logger.info("Checking relevance of search results")
    logger.debug(f"Processing {len(search_results)} results")
    
    prompt = load_prompt("relevance_check")
    logger.debug(f"Loaded prompt template: {prompt}")
    
    prompt_template = ChatPromptTemplate.from_messages([("system", prompt)])
    model = ChatOpenAI().with_structured_output(RelevanceCheckOutput)
    
    chain = prompt_template | model
    logger.debug("Running relevance check chain")
    
    try:
        result = chain.invoke({"input_search_results": json.dumps(search_results, indent=2)})
        logger.info(f"Found {len(result.relevant_results)} relevant results")
        return result
    except Exception as e:
        logger.error(f"Failed to process relevance check: {str(e)}")
        raise


def convert_html_to_markdown(html_content: str) -> str:
    """Convert HTML content to markdown format."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Convert headers
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        level = int(h.name[1])
        h.replace_with('#' * level + ' ' + h.get_text() + '\n\n')
    
    # Convert links
    for a in soup.find_all('a'):
        href = a.get('href', '')
        text = a.get_text()
        if href and text:
            a.replace_with(f'[{text}]({href})')
    
    # Convert formatting
    for tag, marker in [
        (['b', 'strong'], '**'),
        (['i', 'em'], '*')
    ]:
        for element in soup.find_all(tag):
            element.replace_with(f'{marker}{element.get_text()}{marker}')
    
    # Convert lists
    for ul in soup.find_all('ul'):
        for li in ul.find_all('li'):
            li.replace_with(f'- {li.get_text()}\n')
    
    for ol in soup.find_all('ol'):
        for i, li in enumerate(ol.find_all('li'), 1):
            li.replace_with(f'{i}. {li.get_text()}\n')
    
    # Clean up text
    text = soup.get_text()
    return re.sub(r'\n\s*\n', '\n\n', text).strip()


def scrape_and_save_markdown(relevant_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Scrape HTML content from URLs and save as markdown."""
    logger.info(f"Scraping content from {len(relevant_results)} URLs")
    markdown_contents = []
    
    for result in relevant_results:
        url = result.get('link')
        if not url:
            logger.warning(f"No URL found in result: {result}")
            continue
            
        logger.debug(f"Scraping URL: {url}")
        try:
            response = requests.get(
                url,
                headers={'X-API-KEY': os.getenv("SCRAPING_API_KEY")}
            )
            if response.status_code != 200:
                logger.error(f"Failed to scrape {url}: {response.status_code}")
                continue
                
            html_content = response.text
            logger.debug(f"Successfully scraped {len(html_content)} bytes from {url}")
            
            markdown_content = convert_html_to_markdown(html_content)
            logger.debug(f"Converted HTML to {len(markdown_content)} bytes of markdown")
            
            markdown_contents.append({
                'content': markdown_content,
                'url': url,
                'title': result.get('title', '')
            })
            logger.info(f"Successfully processed {url}")
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            continue
    
    logger.info(f"Successfully scraped {len(markdown_contents)} URLs")
    return markdown_contents


def generate_summaries(markdown_contents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Generate summaries for markdown content using gpt-4o."""
    logger.info(f"Generating summaries for {len(markdown_contents)} markdown contents")
    pathlib.Path("markdown_summaries").mkdir(exist_ok=True)
    summary_prompt = load_prompt("summarise_markdown_page")
    summary_template = ChatPromptTemplate.from_messages([("system", summary_prompt)])
    llm = ChatOpenAI(model="gpt-4o")
    summary_chain = summary_template | llm
    
    summaries = []
    for content in markdown_contents:
        try:
            summary = summary_chain.invoke({
                'markdown_input': ' '.join(content['content'].split()[:2000])
            })
            
            summary_filename = f"summary_{content['url']}.md"
            summary_filepath = os.path.join("markdown_summaries", summary_filename)
            
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write(summary.content)
            
            summaries.append({
                'markdown_summary': summary.content,
                'url': content['url']
            })
                
        except Exception as e:
            logger.error(f"Failed to summarize {content['url']}: {str(e)}")

    logger.info(f"Successfully generated {len(summaries)} summaries")
    return summaries


def summariser(state: State) -> Dict:
    """Generate email summary from the state."""
    logger.info("Generating email summary")
    summariser_output = llm_summariser.invoke({
        "messages": state["messages"],
        "list_of_summaries": state["summaries"],
        "input_template": state["email_template"]
    })
    new_messages = [
        AIMessage(content=summariser_output.email_summary),
        AIMessage(content=summariser_output.message)
    ]
    return {
        "messages": new_messages,
        "created_summaries": [summariser_output.email_summary]
    }


def reviewer(state: State) -> Dict:
    """Review the generated summary."""
    logger.info("Reviewing generated summary")
    converted_messages = [
        HumanMessage(content=msg.content) if isinstance(msg, AIMessage)
        else AIMessage(content=msg.content) if isinstance(msg, HumanMessage)
        else msg
        for msg in state["messages"]
    ]
    
    state["messages"] = converted_messages
    reviewer_output = llm_reviewer.invoke({"messages": state["messages"]})
    
    return {
        "messages": [HumanMessage(content=reviewer_output.message)],
        "approved": reviewer_output.approved
    }


def conditional_edge(state: State) -> Literal["summariser", END]:
    """Determine next step based on approval status."""
    logger.info("Determining next step based on approval status")
    return END if state["approved"] else "summariser"


def send_email(email_content: str, destination_email: str):
    """Send email using Sendinblue API."""
    logger.info(f"Sending email to {destination_email}")
    
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key['api-key'] = os.getenv("SENDINGBLUE_API_KEY")
    logger.debug("Configured Sendinblue API client")
    
    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
        sib_api_v3_sdk.ApiClient(configuration)
    )
    
    email_params = {
        "subject": "Daily AI Research Summary",
        "sender": {"name": "Will White", "email": destination_email},
        "html_content": email_content,
        "to": [{"email": destination_email, "name": "Will White"}],
        "params": {"subject": "Daily AI Research Summary"}
    }
    logger.debug(f"Email parameters prepared: {email_params}")
    
    try:
        api_response = api_instance.send_transac_email(
            sib_api_v3_sdk.SendSmtpEmail(**email_params)
        )
        logger.info(f"Email sent successfully: {api_response}")
    except ApiException as e:
        logger.error(f"Failed to send email: {e}")
        raise


def main():
    """Main execution flow."""
    logger.info("Starting email research assistant")
    
    try:
        # Try to load environment variables from .env file first
        if os.path.exists(".env"):
            logger.info("Loading environment variables from .env file")
            with open(".env", "r") as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
            logger.debug("Environment variables loaded from .env file")
        
        # Validate environment variables after loading
        validate_environment_variables()
    except ValueError as e:
        logger.error(f"Environment validation failed: {e}")
        print(f"Error: {e}")
        print("Please ensure all required environment variables are set in the .env file:")
        for var in required_environment_variables:
            print(f"- {var}")
        return
    
    # Get validated destination email
    destination_email = os.getenv("DESTINATION_EMAIL")
    logger.info(f"Using destination email: {destination_email}")
    
    # Search and filter results
    logger.info("Starting search and filtering process")
    relevant_results = []
    for search_term in SEARCH_TERMS:
        logger.info(f"Processing search term: {search_term}")
        results = search_serper(search_term)
        filtered_results = check_search_relevance(results)
        relevant_ids = [r.id for r in filtered_results.relevant_results]
        filtered_results = [r for r in results if r['id'] in relevant_ids]
        relevant_results.extend(filtered_results)
    logger.info(f"Found {len(relevant_results)} total relevant results")
    
    # Process content
    logger.info("Processing content")
    markdown_contents = scrape_and_save_markdown(relevant_results)
    summaries = generate_summaries(markdown_contents)
    logger.info(f"Generated {len(summaries)} summaries")
    
    # Create workflow
    logger.info("Setting up workflow")
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("summariser", summariser)
    workflow.add_node("reviewer", reviewer)
    
    # Add edges
    workflow.add_edge("summariser", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        conditional_edge,
        {
            "continue": "summariser",
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("summariser")
    logger.debug("Workflow graph configured")
    
    # Compile
    logger.info("Compiling workflow")
    app = workflow.compile()
    
    # Run
    logger.info("Running workflow")
    config = {"configurable": {"summaries": summaries}}
    for output in app.stream(config):
        logger.debug(f"Workflow output: {output}")
        if output.get("email_content"):
            logger.info("Email content generated, sending email")
            send_email(output["email_content"], destination_email)
    
    logger.info("Email research assistant completed successfully")


if __name__ == "__main__":
    main()
