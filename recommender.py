import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MusicRecommender:
    def __init__(self):
        """Initializes the music recommender with sample data."""
        self.emotion_map = {
            'happy': ['happy', 'upbeat', 'pop', 'dance', 'summer', 'joy', 'feel-good'],
            'sad': ['sad', 'melancholic', 'chillout', 'ambient', 'acoustic', 'calm'],
            'angry': ['angry', 'metal', 'rock', 'industrial', 'hardcore', 'intense'],
            'fear': ['ambient', 'dark', 'eerie', 'experimental', 'soundtrack'],
            'surprise': ['electronic', 'pop', 'energetic', 'exciting'],
            'neutral': ['lounge', 'jazz', 'instrumental', 'easy listening'],
            'disgust': ['punk', 'grindcore', 'noise', 'protest']
        }
        self._prepare_recommender()
        print("MusicRecommender initialized with sample data.")

    def _prepare_recommender(self):
        """Creates a sample dataframe and builds the TF-IDF matrix."""
        # This sample data contains the necessary 'tags' column.
        song_data = {
            'track_name': ['Queen - Bohemian Rhapsody', 'Adele - Someone Like You', 'Nirvana - Smells Like Teen Spirit', 'Pharrell Williams - Happy', 'Marconi Union - Weightless', 'Metallica - Enter Sandman'],
            'tags': ['rock epic classic', 'sad piano ballad soul', 'rock grunge angry alternative', 'pop happy upbeat feel-good', 'ambient calm relaxing instrumental', 'metal angry intense rock']
        }
        self.df_songs = pd.DataFrame(song_data)
        
        # This uses TF-IDF on tags to analyze text content.
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df_songs['tags'])
            
    def recommend(self, emotion):
        """Recommends songs based on a detected emotion using cosine similarity."""
        emotion = emotion.lower()
        query_tags = ' '.join(self.emotion_map.get(emotion, []))
        
        if not query_tags:
            return []
            
        query_vector = self.tfidf.transform([query_tags])
        cosine_sim = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        related_song_indices = cosine_sim.argsort()[:-6:-1]
        
        return self.df_songs['track_name'].iloc[related_song_indices].tolist()