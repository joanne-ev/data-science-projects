import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import os
    from dotenv import load_dotenv
    from pydantic import BaseModel, Field
    from typing import Literal

    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.common_tools.tavily import tavily_search_tool

    load_dotenv(override=True)
    return (
        Agent,
        BaseModel,
        Field,
        Literal,
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAIProvider,
        os,
        tavily_search_tool,
    )


@app.cell
def _(BaseModel, Field, Literal):
    class heroine_movies(BaseModel):
        movie_list: list[str] = Field(description='List of the names of movies with a superheroine lead. This list only contains movie names.')  

    class BoxOffice(BaseModel):
        name: str
        cinematic_universe: Literal["DC", "Marvel"] = Field(description='Cinematic universe')
        release_year: int = Field(description='Formatted as YYYY', examples=[2026, 2025])
        release_month: str = Field(description='Formatted as its 3-letter short name', examples=['Nov', 'Dec'])
        domestic_opening_week_box_office: float = Field(description='Domestic opening week box office revenue in USD', ge=0)
        domestic_lifetime_box_office: float = Field(description='Total domestic lifetime box office revenue in USD', ge=0)
        worldwide_opening_week_box_office: float = Field(description='Worldwide opening week box office revenue in USD', ge=0)
        worldwide_lifetime_box_office: float = Field(description='Total worldwide lifetime box office revenue in USD (not opening week)', ge=0)

    return BoxOffice, heroine_movies


@app.cell
async def _(
    Agent,
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIProvider,
    heroine_movies,
    os,
    tavily_search_tool,
):
    provider = OpenAIProvider(
        base_url='https://router.huggingface.co/v1',
        api_key=os.getenv('HF_TOKEN'),
    )

    model = OpenAIChatModel(
        'deepseek-ai/DeepSeek-V4-Flash:deepinfra',   # inference providers and cost can be found here: https://huggingface.co/inference/models?model=deepseek-ai/DeepSeek-V4-Flash
        provider=provider,
        settings=OpenAIChatModelSettings(temperature=0),
    )

    agent = Agent(
        model=model,
        output_type=heroine_movies,
        tools=[
            tavily_search_tool(
                api_key=os.getenv('TAVILY_API_KEY'),
                max_results=3,
                include_domains=["screenrant.com", 'wikipedia.com']
            )
        ],

        instructions='''
        Only include theatrical feature films where all primary protagonists are superheroine(s) or women in general.

        Exclude films in which:
        - a male superhero is the primary protagonist, even if a superheroine is a co-lead or appears in the title;
        - the superheroine is part of an ensemble without being a primary protagonist;
        - the film is a TV series or streaming-only release.
        '''
    )

    results = await agent.run('List all movies with a superheroine lead in either Marvel or DC')

    heroine_movies_dict = results.output.model_dump()
    return (heroine_movies_dict,)


@app.cell
def _(heroine_movies_dict):
    import polars as pl 

    pl.DataFrame(heroine_movies_dict)
    return


@app.cell
def _(
    Agent,
    AgentRunResult,
    BoxOffice,
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIProvider,
    os,
    tavily_search_tool,
):
    async def box_office_agent(movie: str) -> AgentRunResult[BoxOffice]:

        bo_provider = OpenAIProvider(
            base_url='https://router.huggingface.co/v1',
            api_key=os.getenv('HF_TOKEN'),
        )

        bo_model = OpenAIChatModel(
            'google/gemma-4-31B-it:deepinfra',  # inference providers can be found here: https://huggingface.co/inference/models?model=google/gemma-4-31B-it
            provider=bo_provider,
            settings=OpenAIChatModelSettings(temperature=0) # reducing randomness
        )

        bo_agent: Agent[object, BoxOffice] = Agent(
            model=bo_model,

            output_type=BoxOffice,

            tools=[
                tavily_search_tool(
                    api_key=os.getenv('TAVILY_API_KEY'), 
                    max_results=5,  # increase the number of results per query allowing the model to cross-reference
                    include_domains=['boxofficemojo.com'],
                )
            ],

            instructions='''
            Use the Tavily search tool to find factual box office data for theatrical films only. 
            Ignore and refuse to search for TV shows, TV specials, or streaming-exclusive series. 
            If a title could refer to either a film or a series, only report on the theatrical film. 
            If no real theatrical film matches the query, say so rather than guessing or fabricating figures.

            Report the precise dollar figures exactly as stated in the source — do not round, estimate, or approximate. 
            If exact figures are unavailable, search again with a more specific query rather than guessing.
            '''
        )

        results: AgentRunResult[BoxOffice] = await bo_agent.run(f'What are the box office revenues for {movie}?')

        return results

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    test.output.model_dump()
    """)
    return


if __name__ == "__main__":
    app.run()
