import streamlit as st
import os
import json
import tempfile
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
    
    text_data = [f"{blog['blog_name'].lower()} {blog['title'].lower()}" for blog in blogs]
    
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

# Initialize session state for input fields
if "blog_name" not in st.session_state:
    st.session_state["blog_name"] = ""
    st.session_state["title"] = ""
    st.session_state["description"] = ""
    st.session_state["image_file"] = None

# Blog Form
st.sidebar.header("📝 Add a New Blog")
st.session_state.blog_name = st.sidebar.text_input("Blog Name", value=st.session_state.blog_name, key="blog_name_input")
st.session_state.title = st.sidebar.text_input("Title", value=st.session_state.title, key="title_input")
st.session_state.image_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="image_file_input")
st.session_state.description = st.sidebar.text_area("Description", value=st.session_state.description, key="description_input")

if st.sidebar.button("Publish Blog"):
    if not st.session_state.blog_name.strip() or not st.session_state.title.strip() or not st.session_state.description.strip() or not st.session_state.image_file:
        st.sidebar.error("⚠️ All fields must be filled before publishing!")
    else:
        image_path = None
        if st.session_state.image_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.image_file.name)[1]) as temp_file:
                temp_file.write(st.session_state.image_file.getbuffer())
                image_path = temp_file.name
        
        new_blog = {"blog_name": st.session_state.blog_name, "title": st.session_state.title, "image": image_path, "description": st.session_state.description}
        add_blog(new_blog)
        st.sidebar.success("✅ Blog Published Successfully!")
        
        # Clear fields
        st.session_state.blog_name = ""
        st.session_state.title = ""
        st.session_state.description = ""
        st.session_state.image_file = None
        st.rerun()

# Blog Display
st.header("📚 All Blogs")
blogs = load_data()

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

if search_query:
    results = search_blogs(search_query, blogs)
    
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
