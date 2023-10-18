import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import base64
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook

# Fungsi untuk crawling data komentar dari video YouTube
def video_comments(api_key, video_id):
    replies = []
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_response = youtube.commentThreads().list(
        part='snippet,replies', videoId=video_id).execute()

    while video_response:
        for item in video_response['items']:
            published = item['snippet']['topLevelComment']['snippet']['publishedAt']
            user = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            replies.append([published, user, comment])

            replycount = item['snippet']['totalReplyCount']
            if replycount > 0:
                for reply in item['replies']['comments']:
                    published = reply['snippet']['publishedAt']
                    user = reply['snippet']['authorDisplayName']
                    repl = reply['snippet']['textDisplay']
                    replies.append([published, user, repl])

        if 'nextPageToken' in video_response:
            video_response = youtube.commentThreads().list(
                part='snippet,replies',
                pageToken=video_response['nextPageToken'],
                videoId=video_id
            ).execute()
        else:
            break

    return replies

# Tampilan aplikasi menggunakan Streamlit
st.title("Crawling YouTube Comments")

api_key = st.text_input("Input API Key YouTube")
video_ids = st.text_area("Input ID Video YouTube (pisahkan dengan koma atau newline)")

if st.button("Crawl Comments"):
    if api_key and video_ids:
        video_ids_list = [vid.strip() for vid in video_ids.replace('\n', ',').split(',')]
        all_comments = []

        for vid in video_ids_list:
            comments = video_comments(api_key, vid)
            all_comments.extend(comments)

        # Menampilkan komentar dalam DataFrame
        df = pd.DataFrame(all_comments, columns=['Create-At', 'Username', 'Comments'])
        st.dataframe(df)

        # Menambahkan tombol download Excel
        st.markdown("<h3>Download Dataset</h3>", unsafe_allow_html=True)
        excel_data = [df.columns.tolist()] + df.values.tolist()
        wb = Workbook()
        ws = wb.active
        for row in excel_data:
            ws.append(row)
        excel_file = "Crawling-Result.xlsx"
        wb.save(excel_file)
        with open(excel_file, "rb") as file:
            b64 = base64.b64encode(file.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{excel_file}">Download Excel</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.text("Please input API Key and ID Video YouTube")