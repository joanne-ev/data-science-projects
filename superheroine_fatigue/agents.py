import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import os
    import csv
    import polars as pl 
    from dotenv import load_dotenv

    from typing import Literal, Optional
    from pydantic import BaseModel, Field

    from pydantic_ai import Agent, AgentRunResult
    from pydantic_ai.common_tools.exa import ExaToolset
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.common_tools.tavily import tavily_search_tool
    from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings

    load_dotenv(override=True)
    return (
        Agent,
        AgentRunResult,
        BaseModel,
        ExaToolset,
        Field,
        Literal,
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAIProvider,
        Optional,
        csv,
        duckduckgo_search_tool,
        os,
        pl,
        tavily_search_tool,
    )


@app.cell
async def _(
    Agent,
    AgentRunResult,
    BaseModel,
    Field,
    HeroineMovies,
    Literal,
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIProvider,
    csv,
    duckduckgo_search_tool,
    os,
):
    # Structured output
    class CinematicUniverse(BaseModel):
        name: str
        cinematic_universe: Literal["DC", "Marvel"] = Field(description='Cinematic universe')
        heroine_lead: Literal['Yes', 'No'] = Field(description='Describes ')

    # AI agent 
    async def cinematic_universe(movie: str):
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
            output_type=CinematicUniverse,
            tools=[duckduckgo_search_tool()],

            instructions='''
            Use the DuckDuckGo search tool to determine 
            - which cinematic universe the movie is a part of and 
            - whether the primary protagonist of the movie has a woman or superheroine 
            '''
        )

        results: AgentRunResult[HeroineMovies] = await agent.run(f'Is {movie} apart of the DC or Marvel cinematic universe and is its main protagonist a woman?')

        return results


    # Running AI agent
    agent = False

    if agent:
        # Create a list of elements from a CSV
        with open('movies.csv') as file:
            movie_reader = csv.reader(file)     # return a reader object which will iterate over lines in the CSV
            next(movie_reader)  # skip header
            title_list = [row[0] for row in movie_reader]   # create one list with all elements 

        title_list = []

        for title in title_list:
            title_result = (await cinematic_universe(title)).output
            title_dict = title_result.model_dump()
            title_list.append(title_dict)
    return


@app.cell
async def _(
    Agent,
    AgentRunResult,
    BaseModel,
    Field,
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIProvider,
    csv,
    os,
    pl,
    tavily_search_tool,
):
    # Structured output
    class HeroineMovies(BaseModel):
        movie_list: list[str] = Field(description='List of the names of movies with a superheroine lead. This list only contains movie names.')  


    # AI agent
    async def superheroine_movies_agent():
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
            output_type=HeroineMovies,
            tools=[
                tavily_search_tool(
                    api_key=os.getenv('TAVILY_API_KEY'),
                    max_results=3,
                    include_domains=["en.wikipedia.org"]
                )
            ],

            instructions='''
            Only include theatrical live-action feature film releases from Marvel or DC where every primary protagonist is female or at least one primary protagonist is a superheroine (or becomes one during the film).

            Exclude films in which:
            - a male superhero is the primary protagonist, even if a superheroine is a co-lead;
            - the superheroine is part of an ensemble without being a primary protagonist;
            - the film is a TV series or streaming-only release.
            '''
        )

        results: AgentRunResult[HeroineMovies] = await agent.run('List all movies with a superheroine lead in either Marvel or DC')

        return results


    # Running AI agent 
    agent1 = False

    if agent1: 
        movies = await superheroine_movies_agent()
        movies_dict = movies.output.model_dump()
        pl.DataFrame(movies_dict).write_csv('movies.csv')

        with open('movies-260704.csv') as file1:
            # Return a reader object which will iterate over lines in the given csvfile.
            reader = csv.reader(file1)
            next(reader)  # skip header
            movie_list = [row[0] for row in reader]
    return HeroineMovies, movie_list


@app.cell
async def _(
    Agent,
    BaseModel,
    ExaToolset,
    Field,
    Literal,
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIProvider,
    Optional,
    csv,
    movie_list,
    os,
):
    # Structured output
    class BoxOffice(BaseModel):
        name: str
        cinematic_universe: Literal["DC", "Marvel"] = Field(description='Cinematic universe')
        release_year: int = Field(description='Formatted as YYYY', examples=[2026, 2025])
        release_month: Literal["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] = Field(description='Formatted as its 3-letter short name')
        domestic_opening_weekend_box_office: float = Field(default=None, description='Domestic opening weekend box office revenue in USD', ge=0)
        worldwide_opening_weekend_box_office: Optional[float] = Field(default=None, description="Worldwide opening weekend box office revenue in USD, if available.")
        domestic_lifetime_box_office: float = Field(default=None, description='Total domestic lifetime box office revenue in USD', ge=0)
        worldwide_lifetime_box_office: float = Field(default=None, description='Total worldwide lifetime box office revenue in USD (not opening weekend)', ge=0)


    # AI agent
    async def box_office_agent(movie: str):

        provider = OpenAIProvider(
            base_url='https://router.huggingface.co/v1',
            api_key=os.getenv('HF_TOKEN'),
        )

        model = OpenAIChatModel(
            'deepseek-ai/DeepSeek-V4-Flash:deepinfra',
            provider=provider,
            settings=OpenAIChatModelSettings(temperature=0) # reducing randomness
        )

        exa_search_tool = ExaToolset(
            api_key=os.getenv('EXA_API_KEY'),
            num_results=5,
            max_characters=1000,  # limit text content to control token usage
            include_get_contents=False
        )

        agent: Agent[object, BoxOffice] = Agent(
            model=model,

            output_type=BoxOffice,

            toolsets=[exa_search_tool],

            retries=3,

            instructions='''
                Use the Exa search tool to find factual box office data for theatrical films only.

                Scope:
                - Only report on theatrical feature films (had a cinema release), not TV shows, TV specials, or streaming-exclusive series.
                - If a title could refer to either a film or a series, only report on the theatrical film.
                - If a title has no theatrical release at all, do not fabricate data — treat all fields as unavailable by setting that field's value to null (None).
                - Every field in the output schema must always be present with either a real figure or null — never omit a field entirely.

                Accuracy rules:
                - Report the precise dollar figures exactly as stated in the source. Do not round, estimate, infer, or calculate missing values from other figures.
                - Prefer sources reporting exact, unrounded figures (e.g. "$80,366,312") over sources reporting rounded figures (e.g. "$80 million"). If only a rounded figure is available after thorough searching, use it.
                - All figures should be in USD unless the source explicitly states another currency. If a source reports a non-USD figure and no USD figure is available, treat the USD field as unavailable.
                - Never copy a value from one field into another (e.g. domestic into worldwide, opening week into lifetime) unless the source explicitly states they are identical for that specific film.
                - If you cannot find a separate international or worldwide figure distinct from the domestic figure, treat the worldwide field as unavailable and set it to null (None). Do not assume a film had no international release just because international box office data was not found in your search results. A missing number is not the same as a zero or a duplicate of another field.
                - If multiple sources disagree on a figure, prefer Box Office Mojo over other sources. If neither is available, use the most recent or most cited figure and do not average conflicting values.

                Search process:
                - If an exact figure for a specific metric is not found on the first search, search again with a more specific query (e.g. adding "box office mojo", the exact film title and year, or the specific metric name) before giving up.
                - If a value cannot be found after multiple search attempts, set that field's value to null (None) rather than guessing.
            '''
        )

        results = await agent.run(f'What are the box office revenues for {movie}?')

        return results


    # Running AI agent
    agent2 = False

    if agent2: 
        box_office_list: list[dict] = []

        for i, movie in enumerate(movie_list):
            box_office = await box_office_agent(movie)
            box_office_dict = box_office.output.model_dump()
            box_office_list.append(box_office_dict)
            print(f'Progress: {i+1}/{len(movie_list)}')

        with open("box-office.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=box_office_list[0].keys())
            writer.writeheader()
            writer.writerows(box_office_list)
    return


if __name__ == "__main__":
    app.run()
