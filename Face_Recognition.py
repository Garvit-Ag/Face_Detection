import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
# cap = cv2.VideoCapture('test.mp4')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y , w ,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0 , 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey ,ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 0, 255), 5)

    # Display the output
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# import warnings
# import os
# import pyttsx3
# import speech_recognition as sr
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain import HuggingFaceHub
#
# # Suppress warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
#
# # Define a custom function to suppress the warning
# def custom_warn_on_import(name: str) -> None:
#     pass
#
# HuggingFaceHub._warn_on_import = custom_warn_on_import
#
# # Initialize the text-to-speech engine
# engine = pyttsx3.init()
# def speak(word, print_only=False):
#     engine.setProperty('rate', 135)
#     engine.setProperty('volume', 0.8)
#     voices = engine.getProperty('voices')
#     engine.setProperty('voice', voices[0].id)
#
#     if print_only:
#         print(word)  # Print links directly
#     else:
#         print(word)  # Print spoken words
#         engine.say(str(word))
#         engine.runAndWait()
#         engine.stop()
#
# # Initialize the speech recognizer
# rec = sr.Recognizer()
# speak("HI ,my name is rocky i am your personal assistant how can i help you?")
# def listen():
#     with sr.Microphone() as source:
#         print("Listening... Speak your query:")
#         audio = rec.listen(source)
#         text = rec.recognize_google(audio)
#         print(f"You said: {text}")
#     return text
#
#
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
#
#
# file_path = "data2.txt"
# encoding = "utf-8"
#
# try:
#     loader = TextLoader(file_path, encoding=encoding)
#     document = loader.load()
# except UnicodeDecodeError:
#     print(f"Unable to decode the file '{file_path}' with encoding: {encoding}")
#
# # Split the document into chunks
# text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
# docs = text_splitter.split_documents(document)
#
# # Embed the text using Hugging Face embeddings
# embedding = HuggingFaceEmbeddings()
#
# # Create a vector store using FAISS
# db = FAISS.from_documents(docs, embedding)
#
# # Load the question-answering chain
# llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 512})
# chain = load_qa_chain(llm, chain_type="stuff")
#
# while True:
#     query = listen()  # Listen to your query
#     doc = db.similarity_search(query)
#
#     # Check if the query is "thank you"
#     if "thank you" in query.lower():
#         speak("You're welcome.")  # Respond to "thank you"
#         break  # Terminate the program
#
#     # Perform question answering
#     question_text = query
#     answers = chain.run(input_documents=doc, question=question_text)
#     speak(answers)
