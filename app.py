import streamlit as st
import os
import json
import base64
import io
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

# 🌟 Constants
DATA_FILE = "blogs.json"

def load_data():
    """Loads blog data from the JSON file."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            pass
    return []

def save_data(data):
    """Saves blog data to the JSON file."""
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def encode_image(image_file):
    """Encodes an uploaded image to a base64 string."""
    return base64.b64encode(image_file.read()).decode("utf-8")

def decode_image(image_data):
    """Decodes a base64 string back into an image."""
    try:
        return Image.open(io.BytesIO(base64.b64decode(image_data)))
    except Exception:
        return None

def add_blog(blog):
    """Adds a new blog post if it does not already exist."""
    data = load_data()
    
    # 🚀 Prevent duplicate blogs
    if any(b["blog_name"].lower() == blog["blog_name"].lower() and b["title"].lower() == blog["title"].lower() for b in data):
        return False  # Blog already exists
    
    data.append(blog)
    save_data(data)
    return True  # Successfully added

def delete_blog(title):
    """Deletes a blog by title."""
    data = load_data()
    save_data([blog for blog in data if blog["title"] != title])

def search_blogs(query, blogs):
    """AI-powered blog search using TF-IDF (title, blog_name, and description)."""
    if not blogs or not query.strip():
        return []
    
    # Include title, blog_name, and description in search
    text_data = [f"{b['blog_name'].lower()} {b['title'].lower()}" for b in blogs]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)
    query_vector = vectorizer.transform([query.lower()])
    
    scores = (tfidf_matrix * query_vector.T).toarray()
    results = sorted(zip(scores, blogs), key=lambda x: -x[0][0])
    
    return [b for score, b in results if score[0] > 0]


# 🎨 Streamlit UI Configuration
st.set_page_config(page_title="📖 Blog Platform", layout="wide")
st.header("Growth Mindset Challenge", divider=True)
st.write("Growth Mindset Challenge is an interactive blogging platform designed to inspire learning, reflection, and personal growth. It allows users to share insightful blogs, upload images, and engage in an AI-powered search to discover inspiring content. Whether you're writing about self-improvement, productivity, or personal experiences, this platform helps cultivate a growth mindset by encouraging continuous learning and sharing valuable insights with the community. 🚀✨")
st.title("📖 Blog Platform")

# ✅ Sidebar: Manage Input Fields with Session State
if "blog_name" not in st.session_state:
    st.session_state.blog_name = ""
if "title" not in st.session_state:
    st.session_state.title = ""
if "description" not in st.session_state:
    st.session_state.description = ""

# ✍️ Sidebar: Add a New Blog
st.sidebar.header("📝 Add a New Blog")
blog_name = st.sidebar.text_input("📌 Blog Name", value=st.session_state.blog_name, key="blog_name_input")
title = st.sidebar.text_input("📰 Title", value=st.session_state.title, key="title_input")
image_file = st.sidebar.file_uploader("📸 Upload Image", type=["png", "jpg", "jpeg"])
description = st.sidebar.text_area("📝 Description", value=st.session_state.description, key="description_input")

if st.sidebar.button("🚀 Publish Blog"):
    if not all([blog_name.strip(), title.strip(), description.strip(), image_file]):
        st.sidebar.error("⚠️ All fields must be filled before publishing!")
    else:
        image_data = encode_image(image_file)
        new_blog = {"blog_name": blog_name, "title": title, "image": image_data, "description": description}
        
        if add_blog(new_blog):
            st.sidebar.success("✅ Blog Published Successfully!")
            
            # 🛠️ Reset the fields (except image uploader)
            st.session_state.blog_name = ""
            st.session_state.title = ""
            st.session_state.description = ""

            st.rerun()  # Refresh the page to clear inputs
        else:
            st.sidebar.warning("⚠️ A blog with the same name and title already exists!")

# 📚 Display All Blogs
st.header("📚 All Blogs")
blogs = load_data()

if not blogs:
    st.info("ℹ️ No blogs available. Add one from the sidebar!")
else:
    for blog in blogs:
        with st.expander(f"📰 {blog['title']}"):
            if blog["image"]:
                img = decode_image(blog["image"])
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.warning("⚠️ Image could not be loaded.")
            st.write(f"📝 {blog['description']}")
            st.caption(f"✍️ {blog['blog_name']}")
            if st.button(f"🗑️ Delete '{blog['title']}'", key=f"delete_{blog['title']}"):
                delete_blog(blog['title'])
                st.rerun()

# 🔍 AI-Powered Search
st.header("🔍 AI-Powered Search")
search_query = st.text_input("🔎 Search Blogs by Content")

if search_query:
    results = search_blogs(search_query, blogs)
    
    if results:
        st.subheader("🔎 Search Results")
        for res in results:
            with st.expander(f"📰 {res['title']}"):
                if res["image"]:
                    img = decode_image(res["image"])
                    if img:
                        st.image(img, use_container_width=True)
                    else:
                        st.warning("⚠️ Image could not be loaded.")
                st.write(f"📝 {res['description']}")
                st.caption(f"✍️ {res['blog_name']}")
    else:
        st.warning("⚠️ No matching blogs found!")
