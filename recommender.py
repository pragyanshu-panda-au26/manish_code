import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class HotelRecommender:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.hotels_df = pd.read_excel(os.path.join(current_dir, 'hotels.xlsx'))
        self.users_df = pd.read_excel(os.path.join(current_dir, 'users.xlsx'))
        self._prepare_data()
        
    def _prepare_data(self):
        # Create user-hotel matrix
        self.user_hotel_matrix = pd.pivot_table(
            self.hotels_df,
            values='total',
            index='userCode',
            columns='name',
            aggfunc='sum',
            fill_value=0
        )
        
        # Calculate user similarity matrix
        self.user_similarity = cosine_similarity(self.user_hotel_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_hotel_matrix.index,
            columns=self.user_hotel_matrix.index
        )
    
    def get_user_favorite_hotels(self, user_code, n=3):
        """Get user's most visited hotels based on total spending"""
        user_hotels = self.hotels_df[self.hotels_df['userCode'] == user_code]
        if user_hotels.empty:
            return []
        
        # Group by hotel name and sum total spending
        favorite_hotels = user_hotels.groupby('name')['total'].sum().sort_values(ascending=False)
        return [(hotel, score) for hotel, score in favorite_hotels.head(n).items()]
    
    def get_recommendations(self, user_code, n_recommendations=5):
        if user_code not in self.user_similarity_df.index:
            return []
        
        # Find similar users
        similar_users = self.user_similarity_df.loc[user_code].sort_values(ascending=False)[1:11]
        
        # Get hotels visited by similar users with their scores
        hotel_scores = defaultdict(float)
        user_hotels = set(self.hotels_df[self.hotels_df['userCode'] == user_code]['name'])
        
        # Get user's favorite hotels first
        favorite_hotels = self.get_user_favorite_hotels(user_code)
        recommendations = []
        
        # Add favorite hotels with high confidence score
        for hotel, score in favorite_hotels:
            if len(recommendations) < n_recommendations:
                confidence = min(0.95, score / self.hotels_df['total'].max())  # Normalize score
                recommendations.append((hotel, confidence, "Based on your previous visits"))
        
        # Calculate scores for other hotels based on similar users
        for similar_user, similarity_score in similar_users.items():
            similar_user_hotels = self.hotels_df[self.hotels_df['userCode'] == similar_user]
            for _, row in similar_user_hotels.iterrows():
                hotel = row['name']
                if hotel not in user_hotels and hotel not in [h[0] for h in recommendations]:
                    # Weight the score by both similarity and total spending
                    hotel_scores[hotel] += similarity_score * (row['total'] / self.hotels_df['total'].max())
        
        # Sort hotels by score and add top recommendations
        sorted_hotels = sorted(hotel_scores.items(), key=lambda x: x[1], reverse=True)
        for hotel, score in sorted_hotels:
            if len(recommendations) < n_recommendations and hotel not in [h[0] for h in recommendations]:
                confidence = min(0.9, score)  # Cap confidence at 0.9 for similar user recommendations
                recommendations.append((hotel, confidence, "Based on similar users' preferences"))
        
        return recommendations[:n_recommendations]
