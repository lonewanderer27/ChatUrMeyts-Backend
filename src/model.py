import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
from .supabase import (
    get_all_student_profiles,
    get_group_members,
    get_group_metadata
)
from typing import Dict, Any, List

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroupRecommender:
    def __init__(self):
        self.top_k_groups = 10 # Default Top K groups
        # Initialize placeholders
        self.group_chats = None
        self.group_chats_data = None
        self.group_members = None
        self.group_members_data = None
        self.students = None
        self.students_data = None
        self.interaction_matrix = None
        self.tfidf_matrix = None
        self.tfidf_similarity = None
        self.cf_similarity = None
        self.year_level_similarity = None
        self.combined_similarity = None
        self.distance_matrix = None
        self.hybrid_knn = None

    def load_data(self):
        # Fetch data using RPC functions
        logger.info("Fetching group metadata...")
        self.group_chats_data = get_group_metadata()
        
        logger.info("Fetching group members...")
        self.group_members_data = get_group_members()
        
        logger.info("Fetching student profiles...")
        self.students_data = get_all_student_profiles()

        # Convert to DataFrames
        logger.info("Converting data to DataFrames...")
        self.group_chats = pd.DataFrame(self.group_chats_data)
        self.group_members = pd.DataFrame(self.group_members_data)
        self.students = pd.DataFrame(self.students_data)

        # Handle missing values
        logger.info("Handling missing values...")
        self.students['subjects'] = self.students['subjects'].fillna('')
        self.students['hobbies'] = self.students['hobbies'].fillna('')
        self.students['block'] = self.students['block'].fillna('')  # Ensure column name matches

        self.group_chats['block'] = self.group_chats['block'].fillna('')

        # Prepare student profiles for Content-Based Filtering (CBF)
        logger.info("Preparing student profiles for CBF...")
        self.students['profile_features'] = (
            self.students['subjects'] + ' ' + self.students['hobbies'] + ' ' + self.students['block']
        )

    def prepare_interaction_matrix(self):
        logger.info("Preparing interaction matrix...")
        self.interaction_matrix = self.group_members.pivot_table(
            index='student_id', columns='group_id', aggfunc='size', fill_value=0
        )
        
        # Ensure 'student_id' is of type int64 after pivoting
        self.interaction_matrix.index = self.interaction_matrix.index.astype(np.int64)
        
        # Prepare interaction matrix for Collaborative Filtering (CF)
        self.interaction_matrix = self.interaction_matrix.reindex(index=self.students['id'], fill_value=0)

    def compute_similarities(self):
        logger.info("Computing TF-IDF similarities...")
        tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.students['profile_features'])
        
        # Compute TF-IDF similarity matrix and adjust to [0, 1]
        self.tfidf_similarity = cosine_similarity(self.tfidf_matrix)
        self.tfidf_similarity = (self.tfidf_similarity + 1) / 2  # Adjust to [0, 1]
        
        logger.info("Computing Collaborative Filtering (CF) similarities...")
        # Compute CF similarity matrix and adjust to [0, 1]
        self.cf_similarity = cosine_similarity(self.interaction_matrix)
        self.cf_similarity = (self.cf_similarity + 1) / 2  # Adjust to [0, 1]
        
        # Fill missing values in group_chats and process year_level
        logger.info("Processing year levels...")
        self.group_chats['year_level'] = self.group_chats['year_level'].fillna('N/A')
        self.group_chats['year_level'] = self.group_chats['year_level'].apply(
            lambda x: list(map(int, x.split(','))) if x != 'N/A' else []
        )
        
        # Calculate year level similarity matrix
        logger.info("Calculating year level similarity matrix...")
        self.year_level_similarity = self.year_level_similarity_matrix()
        
        # Combine similarities with weights
        year_level_weight = 0.2
        cf_weight = 0.5
        tfidf_weight = 0.3
        
        logger.info("Combining similarities with weights...")
        self.combined_similarity = (
            (year_level_weight * self.year_level_similarity) +
            (cf_weight * self.cf_similarity) +
            (tfidf_weight * self.tfidf_similarity)
        )
        
        # Ensure combined_similarity is within [0, 1]
        self.combined_similarity = np.clip(self.combined_similarity, 0, 1)
        
        # Compute the distance matrix
        self.distance_matrix = 1 - self.combined_similarity  # Distances are in [0, 1]

    def year_level_similarity_matrix(self):
        logger.info("Generating year level similarity matrix...")
        num_students = len(self.students)
        year_sim_matrix = np.zeros((num_students, num_students))
    
        # Map student IDs to their group IDs
        student_groups = self.group_members.groupby('student_id')['group_id'].apply(list).to_dict()
    
        # Pre-compute year levels for each student based on their group memberships
        student_year_levels = {}
        for student_id in self.students['id']:
            groups = student_groups.get(student_id, [])
            years = []
            for group_id in groups:
                group_years = self.group_chats[self.group_chats['group_id'] == group_id]['year_level']
                if not group_years.empty:
                    years.extend(group_years.iloc[0])
            student_year_levels[student_id] = years
    
        # Compute the similarity matrix
        for i, student1 in self.students.iterrows():
            years1 = student_year_levels.get(student1['id'], [])
            for j, student2 in self.students.iterrows():
                years2 = student_year_levels.get(student2['id'], [])
                similarity = self.compute_year_level_similarity(years1, years2)
                year_sim_matrix[i, j] = similarity
    
        return year_sim_matrix

    @staticmethod
    def compute_year_level_similarity(years1, years2):
        if not years1 or not years2:
            return 0.0
        intersection = len(set(years1).intersection(years2))
        union = len(set(years1).union(years2))
        return intersection / union if union != 0 else 0.0

    def knn_model(self):
        logger.info("Initializing K-NearestNeighbors model...")
        optimal_k = 9
        n_neighbors = optimal_k
        
        # Initialize the NearestNeighbors model with precomputed distances
        self.hybrid_knn = NearestNeighbors(metric='precomputed')
        self.hybrid_knn.fit(self.distance_matrix)

    def initialize(self):
        logger.info("Initializing GroupRecommender...")
        self.load_data()
        self.prepare_interaction_matrix()
        self.compute_similarities()
        self.knn_model()
        logger.info("GroupRecommender initialized successfully.")
    
    def refresh_data(self):
        logger.info("Refreshing data...")
        self.load_data()
        self.prepare_interaction_matrix()
        self.compute_similarities()
        self.knn_model()
        logger.info("Data refreshed successfully.")

    def get_group_members_by_group_id(self, group_id: int):
        """
        Fetch the group members for the given group ID.
        :param group_id: Group ID to fetch members for.
        :return: List of group members dictionaries.
        """
        logger.info(f"Fetching group members for group ID: {group_id}")

        if not isinstance(group_id, int):
            raise ValueError("group_id must be an integer.")

        # Filter group members based on the provided group ID
        # filtered_members = self.group_members[self.group_members['group_id'] == group_id]

        # Filter group members and related student data based on the provided group ID
        filtered_members = self.group_members[self.group_members['group_id'] == group_id]
        filtered_members = filtered_members.merge(self.students, left_on='student_id', right_on='id')

        if filtered_members.empty:
            logger.info(f"No member data found for the provided group ID: {group_id}")
            return []

        # Convert DataFrame rows to dictionaries
        group_members_data = filtered_members.to_dict(orient='records')

        logger.info(f"Fetched {len(group_members_data)} group member records.")

        return group_members_data

    def get_group_chats_by_ids(self, group_ids: List[int]):
        """
        Fetch raw group chat data for the given group IDs.
        :param group_ids: List of group IDs to fetch chats for.
        :return: List of raw group chat data dictionaries.
        """
        logger.info(f"Fetching group chat data for group IDs: {group_ids}")

        if not isinstance(group_ids, list):
            raise ValueError("group_ids must be a list of integers.")

        # Filter group chats based on the provided group IDs
        filtered_chats = self.group_chats[self.group_chats['group_id'].isin(group_ids)]

        if filtered_chats.empty:
            logger.info(f"No chat data found for the provided group IDs: {group_ids}")
            return []

        # Convert DataFrame rows to dictionaries
        raw_chat_data = filtered_chats.to_dict(orient='records')

        logger.info(f"Fetched {len(raw_chat_data)} group chat records.")
        return raw_chat_data

    def recommend_groups_for_student(self, student_id, top_k = None):
        logger.info(f"student_id: {student_id}")
        logger.info(f"top_k: {top_k}")

        if top_k == None:
            top_k = self.top_k_groups

        logger.info(f"Recommend groups for student {student_id}...")

        # Find the index of the student in the dataframe
        student_indices = self.students.index[self.students['id'] == student_id].tolist()
        logger.info(f"student_indices: {student_indices}")

        if not student_indices:
            logger.warning(f"Student ID {student_id} not found.")
            return []
        student_idx = student_indices[0]
       

        logger.info(f"student_idx: {student_idx}")
    
        # Get the distance row for the student
        test_distance_row = self.distance_matrix[student_idx].reshape(1, -1)
    
        # Find nearest neighbors
        distances, indices = self.hybrid_knn.kneighbors(test_distance_row, n_neighbors=9)
        neighbor_indices = indices.flatten()
        neighbor_distances = distances.flatten()
        neighbor_student_ids = self.students.iloc[neighbor_indices]['id'].values
    
        # Get the groups the student is already a member of
        current_groups = self.group_members[self.group_members['student_id'] == student_id]['group_id'].unique()

        logger.info(f"current_groups: {current_groups}")
    
        # Fuzzy KNN: Compute weights (membership degrees)
        epsilon = 1e-5
        weights = 1 / (neighbor_distances + epsilon)
    
        # Get predicted scores for groups
        group_scores = {}
        for neighbor_id, weight in zip(neighbor_student_ids, weights):
            neighbor_groups = self.group_members[self.group_members['student_id'] == neighbor_id]['group_id']
            for group_id in neighbor_groups:
                if group_id not in current_groups:  # Exclude current groups
                    group_scores[group_id] = group_scores.get(group_id, 0) + weight
    
        if not group_scores:
            logger.info(f"No new groups to recommend for student {student_id}.")
            return []

        logger.info(f"group_scores: {group_scores}")
    
        # Sort groups by score
        sorted_groups = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
        recommended_groups = [group_id for group_id, score in sorted_groups[:top_k]]

        logger.info(f"Recommended groups for student {student_id}: {recommended_groups}")
        return recommended_groups
