# Question-Answering System with TF-IDF and Cosine Similarity

This repository contains a Python-based question-answering system that utilizes the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique and cosine similarity to provide answers to user questions. The system can be used in various applications such as chatbots, customer support, and information retrieval.

## Features

- **Natural Language Processing (NLP):** The system preprocesses user queries and stored questions using NLP techniques, including tokenization, stemming, and lemmatization.

- **TF-IDF Vectorization:** It employs TF-IDF vectorization to convert text data into numerical vectors, which helps in capturing the importance of words in documents.

- **Cosine Similarity:** The system calculates cosine similarity between the user query and a database of questions to identify the most relevant questions.

- **Response Selection:** Once the relevant questions are identified, the system selects the best response based on cosine similarity scores.

## Getting Started

1. **Prerequisites:** Make sure you have Python and the required libraries (such as NLTK and scikit-learn) installed. You can install them using `pip`.

2. **Data Preparation:** Prepare your question-answer data in a format similar to the provided `dialogs.txt` file with questions and corresponding answers.

3. **Data Preprocessing:** Customize the preprocessing functions according to your data and requirements in the `qa_app.py` file.

4. **Run the Streamlit App:** Execute the Streamlit app using the `streamlit run qa_app.py` command to interact with the question-answering system.

## Usage

- Input a question in the provided text box, and the system will provide an answer based on the most similar questions in the dataset.

- The system considers questions with a cosine similarity score greater than 0.6 as relevant.

- If no relevant questions are found, the system will respond with a default message.

## Customization

You can customize the system by adjusting the preprocessing functions, TF-IDF parameters, and similarity threshold in the `qa_app.py` file to better suit your specific use case and dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was inspired by the need for a simple question-answering system using NLP techniques.

---

Feel free to add more details, usage examples, or any additional sections that you think would be helpful for users of your GitHub repository.
