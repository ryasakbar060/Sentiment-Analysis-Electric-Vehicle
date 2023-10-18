import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Load model TF-IDF vectorizer, model Naive Bayes, dan model SVM
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vec_loaded = pickle.load(f)

with open('naive_bayes_tfidf.pkl', 'rb') as f:
    modelNBC_loaded = pickle.load(f)

with open('count_vectorizer.pkl', 'rb') as f:
    count_vec_loaded = pickle.load(f)

with open('naive_bayes_vec.pkl', 'rb') as f:
    modelNBC_vec_loaded = pickle.load(f)


def predict_sentiment(comment_text, method):
    if method == 'Naive Bayes With Tf-Idf':
        test_tfidf_load = tfidf_vec_loaded.transform([comment_text]).toarray()
        predicted_label = modelNBC_loaded.predict(test_tfidf_load)[0]
    else:  # Naive Bayes Without Tf-Idf
        test_count_load = count_vec_loaded.transform([comment_text]).toarray()
        predicted_label = modelNBC_vec_loaded.predict(test_count_load)[0]

    return 'Positif' if predicted_label == 1 else 'Negatif'


def main():
    st.title("Sentiment Analysis App")

    method = st.radio("Pilih metode klasifikasi:",
                      ('Naive Bayes With Tf-Idf', 'Naive Bayes Without Tf-Idf'))
    uploaded_file = st.file_uploader(
        "Upload file dataset Excel untuk melakukan analisis sentimen:", type=["xlsx"])

    if uploaded_file is not None:
        test_df_load = pd.read_excel(uploaded_file)
        comment_text = test_df_load['Comment']

        if method == 'Naive Bayes With Tf-Idf':
            test_tfidf_load = tfidf_vec_loaded.transform(
                comment_text).toarray()
            st.write("Ekstraksi Fitur dengan TF-IDF:")
            df_test_tfidf_load = pd.DataFrame(
                test_tfidf_load, columns=tfidf_vec_loaded.get_feature_names_out())
            st.dataframe(df_test_tfidf_load)
        else:
            test_count_load = count_vec_loaded.transform(
                comment_text).toarray()
            st.write("Ekstraksi Fitur tanpa TF-IDF:")
            df_test_vec_load = pd.DataFrame(
                test_count_load, columns=count_vec_loaded.get_feature_names_out())
            st.dataframe(df_test_vec_load)

        # Predict using the selected method
        predicted_labels = [predict_sentiment(
            text, method) for text in comment_text]

        # Membuat DataFrame hasil prediksi
        result_df = pd.DataFrame({
            'Comment': comment_text,
            'Label': test_df_load['Label'],
            'Classification': predicted_labels
        })

        st.write(f"Hasil Analisis Sentimen dengan {method}:")
        st.write(result_df)

        # Plot Grafik Hasil Analisis
        s = pd.value_counts(result_df['Classification'])
        fig, ax = plt.subplots()
        ax.bar(s.index, s)
        n = len(result_df.index)
        for p in ax.patches:
            ax.annotate(str(round(p.get_height() / n * 100, 2)) +
                        '%', (p.get_x() * 1.005, p.get_height() * 1.005))

        plt.xlabel('Kategori')
        plt.ylabel('Jumlah')
        plt.title(f'Hasil Analisis Sentimen dengan {method}')
        st.pyplot(fig)

        # Menghitung akurasi
        label_mapping = {'Negatif': 0, 'Positif': 1}
        test_labels = test_df_load['Label'].map(label_mapping)
        accuracy = accuracy_score(
            test_labels, [1 if label == 'Positif' else 0 for label in predicted_labels])
        st.write(f"Akurasi dengan {method}:", accuracy)

        # Menampilkan classification report
        st.write(f"Classification Report dengan {method}:")
        st.table(pd.DataFrame(classification_report(test_labels, [1 if label == 'Positif' else 0 for label in predicted_labels],
                                                    target_names=['Negatif', 'Positif'], output_dict=True)).transpose())

        # Menampilkan confusion matrix
        matrix = confusion_matrix(
            test_labels, [1 if label == 'Positif' else 0 for label in predicted_labels])
        st.write(f"Confusion Matrix dengan {method}:")
        # st.write(matrix)

        # Visualisasi confusion matrix
        labels = np.unique(test_df_load['Label'])
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix dengan {method}')
        st.pyplot(fig)


if __name__ == "__main__":
    main()
