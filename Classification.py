import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model TF-IDF vectorizer, model Naive Bayes, dan model SVM
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vec_loaded = pickle.load(f)

with open('naive_bayes.pkl', 'rb') as f:
    modelNBC_loaded = pickle.load(f)

with open('support_vector_machine.pkl', 'rb') as f:
    modelSVM_loaded = pickle.load(f)

def predict_sentiment(comment_text):
    test_tfidf_load = tfidf_vec_loaded.transform([comment_text]).toarray()
    predictNBC_load = modelNBC_loaded.predict(test_tfidf_load)
    predictSVM_load = modelSVM_loaded.predict(test_tfidf_load)
    return {
        'Naive Bayes': 'Positif' if predictNBC_load[0] == 1 else 'Negatif',
        'SVM': 'Positif' if predictSVM_load[0] == 1 else 'Negatif'
    }

def main():
    st.title("Sentiment Analysis App")
    st.write("Upload file dataset Excel untuk melakukan analisis sentimen")

    uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx"])

    if uploaded_file is not None:
        test_df_load = pd.read_excel(uploaded_file)

        # Mengonversi teks pada data test ke dalam representasi TF-IDF
        comment_text = test_df_load['comment']
        test_tfidf_load = tfidf_vec_loaded.transform(comment_text).toarray()

        # ----------- NAIVE BAYES -------------
        st.write('NAIVE BAYES')
        predictNBC_load = modelNBC_loaded.predict(test_tfidf_load)

        # Mengubah representasi angka kembali ke label asli
        predicted_labels_NBC_load = pd.Series(predictNBC_load).map({0: 'Negatif', 1: 'Positif'})

        # Membuat DataFrame hasil prediksi
        result_df_NBC_load = pd.DataFrame({
            'Comment': comment_text,
            'Label': test_df_load['label'],
            'Classification': predicted_labels_NBC_load
        })

        st.write("Hasil Analisis Sentimen:")
        st.write(result_df_NBC_load)

        # Plot Grafik Hasil Analisis Naive Bayes
        s = pd.value_counts(result_df_NBC_load['Classification'])
        fig, ax = plt.subplots()
        ax.bar(s.index, s)
        n = len(result_df_NBC_load.index)
        for p in ax.patches:
            ax.annotate(str(round(p.get_height() / n * 100, 2)) + '%', (p.get_x() * 1.005, p.get_height() * 1.005))

        plt.xlabel('Kategori')
        plt.ylabel('Jumlah')
        plt.title('Hasil Analisis Sentimen')
        st.pyplot(fig)

        # Menghitung akurasi
        label_mapping = {'Negatif': 0, 'Positif': 1}
        test_labels = test_df_load['label'].map(label_mapping)
        accuracy = accuracy_score(test_labels, predictNBC_load)
        st.write("Akurasi:", accuracy)

        # Menampilkan classification report
        st.write("Classification Report:")
        st.table(pd.DataFrame(classification_report(test_labels, predictNBC_load, target_names=['Negatif', 'Positif'], output_dict=True)).transpose())

        # Menampilkan confusion matrix
        matrix = confusion_matrix(test_labels, predictNBC_load)
        st.write("Confusion Matrix:")
        st.write(matrix)

        # Visualisasi confusion matrix
        labels = np.unique(test_df_load['label'])
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        st.pyplot(fig)

        # ----------- SVM -------------
        st.write('SVM')
        predictSVM_load = modelSVM_loaded.predict(test_tfidf_load)

        # Mengubah representasi angka kembali ke label asli
        predicted_labels_SVM_load = pd.Series(predictSVM_load).map({0: 'Negatif', 1: 'Positif'})

        # Membuat DataFrame hasil prediksi SVM
        result_df_SVM_load = pd.DataFrame({
            'Comment': comment_text,
            'Label': test_df_load['label'],
            'Classification': predicted_labels_SVM_load
        })

        st.write("Hasil Analisis Sentimen SVM:")
        st.write(result_df_SVM_load)

        # Plot Grafik Hasil Analisis SVM
        s_svm = pd.value_counts(result_df_SVM_load['Classification'])
        fig_svm, ax_svm = plt.subplots()
        ax_svm.bar(s_svm.index, s_svm)
        n_svm = len(result_df_SVM_load.index)
        for p_svm in ax_svm.patches:
            ax_svm.annotate(str(round(p_svm.get_height() / n_svm * 100, 2)) + '%', (p_svm.get_x() * 1.005, p_svm.get_height() * 1.005))

        plt.xlabel('Kategori')
        plt.ylabel('Jumlah')
        plt.title('Hasil Analisis Sentimen SVM')
        st.pyplot(fig_svm)

        # Menghitung akurasi SVM
        accuracy_svm = accuracy_score(test_labels, predictSVM_load)
        st.write("Akurasi SVM:", accuracy_svm)

        # Menampilkan classification report SVM
        st.write("Classification Report SVM:")
        st.table(pd.DataFrame(classification_report(test_labels, predictSVM_load, target_names=['Negatif', 'Positif'], output_dict=True)).transpose())

        # Menampilkan confusion matrix SVM
        matrix_svm = confusion_matrix(test_labels, predictSVM_load)
        st.write("Confusion Matrix SVM:")
        st.write(matrix_svm)

        # Visualisasi confusion matrix SVM
        sns.heatmap(matrix_svm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix SVM')
        st.pyplot(fig_svm)
        


if __name__ == "__main__":
    main()
