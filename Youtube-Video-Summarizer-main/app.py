import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from textblob import TextBlob
import re
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def summarize_text(text):
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

    words = text.split()
    chunk_size = 500
    summaries = []

    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        result = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(result[0]['summary_text'])

    final_summary = ' '.join(summaries)
    return final_summary

def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
    keywords = [word for word in words if word not in stop_words and len(word) > 1]

    counter = CountVectorizer().fit_transform([' '.join(keywords)])
    vocabulary = CountVectorizer().fit([' '.join(keywords)]).vocabulary_
    top_keywords = sorted(vocabulary, key=vocabulary.get, reverse=True)[:5]

    return top_keywords

def split_into_chunks(text, chunk_size=1000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def topic_modeling(text):
    chunks = split_into_chunks(text, chunk_size=1000)
    vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words='english')
    tf = vectorizer.fit_transform(chunks)
    lda_model = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online', random_state=42)
    lda_model.fit(tf)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-6:-1]])
    return topics

def extract_video_id(url):
    video_id = None
    patterns = [
        r'v=([^&]+)',
        r'youtu.be/([^?]+)',
        r'youtube.com/embed/([^?]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break
    return video_id

def main():
    st.title("📺 YouTube Video Summarizer with NLP 🔍")

    video_url = st.text_input("🎥 Enter YouTube Video URL:")

    if st.button("✨ Summarize Video"):
        try:
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("❌ Invalid YouTube URL. Please enter a valid one.")
                return

            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                if not transcript or len(transcript) == 0:
                    st.error("❌ Transcript not available or empty.")
                    return
            except TranscriptsDisabled:
                st.error("❌ Transcripts are disabled for this video.")
                return
            except NoTranscriptFound:
                st.error("❌ No transcript found for this video.")
                return
            except Exception as e:
                st.error(f"🚨 Error fetching transcript: {e}")
                return

            video_text = ' '.join([line['text'] for line in transcript])

            st.info("📄 Summarizing transcript...")
            summary = summarize_text(video_text)

            st.info("🏷️ Extracting keywords...")
            keywords = extract_keywords(video_text)

            st.info("📚 Detecting topics...")
            topics = topic_modeling(video_text)

            sentiment = TextBlob(video_text).sentiment

            st.subheader("📝 Video Summary:")
            st.write(summary)

            st.subheader("🔑 Top Keywords:")
            st.write(keywords)

            st.subheader("📌 Topics:")
            for idx, topic in enumerate(topics):
                st.write(f"Topic {idx + 1}: {', '.join(topic)}")

            st.subheader("📈 Sentiment Analysis:")
            st.write(f"Polarity: {sentiment.polarity:.2f}")
            st.write(f"Subjectivity: {sentiment.subjectivity:.2f}")

        except Exception as e:
            st.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
