from langchain_core.runnables import RunnableSerializable

chain: RunnableSerializable = (
    print
)

chain.invoke({})