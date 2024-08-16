import dspy

from core import get_llm, get_embeddings_model, get_vectorstore

embeddings_model = get_embeddings_model()

vectorstore = get_vectorstore(embeddings_model)

dspy.settings.configure(
    lm=get_llm(),                # Language model
    rm=vectorstore.as_retriever()           # Retriever model
)


# Defining a class named GenerateAnswer which inherits from dspy.Signature
class GenerateAnswer(dspy.Signature):
    """Think and Answer questions based on the context provided."""

    # Defining input fields with descriptions
    context = dspy.InputField(desc="May contain relevant facts about user query")
    question = dspy.InputField(desc="User query")

    # Defining output field with description
    answer = dspy.OutputField(desc="Answer in one or two lines")


# Define a class named RAG inheriting from dspy.Module
class RAG(dspy.Module):
    # Initialize the RAG class
    def __init__(self, num_passages=3):
        # Call the superclass's constructor
        super().__init__()

        # Initialize the retrieve module
        self.retrieve = dspy.Retrieve(k=num_passages)

        # Initialize the generate_answer module using ChainOfThought with GenerateAnswer
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    # Define the forward method
    def forward(self, question):
        # Retrieve relevant context passages based on the input question
        context = self.retrieve(question).passages

        # Generate an answer based on the retrieved context and the input question
        prediction = self.generate_answer(context=context, question=question)

        # Return the prediction as a dspy.Prediction object containing context and answer
        return dspy.Prediction(context=context, answer=prediction.answer)

