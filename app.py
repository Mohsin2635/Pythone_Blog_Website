import streamlit as st
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure uploads folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Data Storage File
DATA_FILE = "blogs.json"

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as file:
                data = json.load(file)
                if isinstance(data, list):
                    return data
        except json.JSONDecodeError:
            pass
    
    return []

def save_data(data):
    with open(DATA_FILE, "w") as file:
        json.dump(data, file, indent=4)

def add_blog(blog):
    data = load_data()
    data.append(blog)
    save_data(data)

def delete_blog(title):
    data = load_data()
    updated_data = [blog for blog in data if blog["title"] != title]
    
    # Delete image file if it exists
    for blog in data:
        if blog["title"] == title and blog["image"]:
            try:
                os.remove(blog["image"])
            except FileNotFoundError:
                pass

    save_data(updated_data)

def search_blogs(query, blogs):
    if not blogs or not query.strip():
        return []
    
    # Combine blog_name and title for better search accuracy
    text_data = [f"{blog['blog_name'].lower().strip()} {blog['title'].lower().strip()}" for blog in blogs]
    
    if len(text_data) < 2:
        return [blog for blog in blogs if query.lower() in blog["blog_name"].lower() or query.lower() in blog["title"].lower()]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)
    query_vector = vectorizer.transform([query.lower()])
    
    scores = (tfidf_matrix * query_vector.T).toarray()
    results = sorted(zip(scores, blogs), key=lambda x: -x[0][0])
    
    return [blog for score, blog in results if score[0] > 0]


# Streamlit UI
st.set_page_config(page_title="AI-Powered Blog Platform", layout="wide")
st.title("📖 Blog Platform")

# Blog Form
st.sidebar.header("📝 Add a New Blog")
blog_name = st.sidebar.text_input("Blog Name")
title = st.sidebar.text_input("Title")
image_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
description = st.sidebar.text_area("Description")

if st.sidebar.button("Publish Blog"):
    if not blog_name or not title or not description:
        st.sidebar.error("⚠️ Please fill in all fields before publishing.")
    else:
        image_path = None
        if image_file:
            image_path = os.path.join(UPLOAD_FOLDER, image_file.name)
            with open(image_path, "wb") as f:
                f.write(image_file.getbuffer())
        
        new_blog = {"blog_name": blog_name, "title": title, "image": image_path, "description": description}
        add_blog(new_blog)
        st.sidebar.success("✅ Blog Published Successfully!")
        st.rerun()


# Blog Display
st.header("📚 All Blogs")
blogs = load_data()
# print(blogs)

if not blogs:
    st.info("No blogs available. Add one from the sidebar!")
else:
    for blog in blogs:
        with st.expander(blog["title"]):
            if blog["image"]:
                st.image(blog["image"], use_container_width=True)
            st.write(blog["description"])
            st.caption(f"✍️ {blog['blog_name']}")
            if st.button(f"🗑️ Delete '{blog['title']}'", key=f"delete_{blog['title']}"):
                delete_blog(blog['title'])
                st.rerun()


# AI Search
st.header("🔍 AI-Powered Search")
search_query = st.text_input("Search Blogs by Content")
# print(search_query)
if search_query:
    results = search_blogs(search_query, blogs)
    # print(results)
    if results:
        st.subheader("Search Results")
        for res in results:
            with st.expander(res["title"]):
                if res["image"]:
                    st.image(res["image"], use_container_width=True)
                st.write(res["description"])
                st.caption(f"✍️ {res['blog_name']}")
    else:
        st.warning("⚠️ No matching blogs found!")
