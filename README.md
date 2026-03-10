# AI Chatbot on Sericulture

## 📌 Overview

This project presents an **AI-based Sericulture Chatbot** designed to provide accurate and real-time advisory support for sericulture farmers. The system uses a **lightweight Large Language Model (LLM)** to understand natural language queries and generate contextually relevant responses based on domain-specific knowledge.

The chatbot consolidates verified sericulture information into a single intelligent advisory platform, reducing reliance on scattered resources and helping farmers make informed decisions.

---

## 🚀 Key Features

* 💬 Natural language query understanding
* 🔎 Retrieval-Augmented Generation (RAG) architecture
* 🧠 Sentence Transformer–based semantic embeddings
* 📚 Curated sericulture knowledge base
* 🌐 Simple and user-friendly web interface
* 📡 Optimized for low-connectivity environments
* 📊 Semantic similarity–based performance evaluation

---

## 🏗️ System Architecture

The chatbot follows a **Retrieval-Augmented Generation (RAG)** pipeline:

### 1️⃣ Knowledge Base Creation

Domain-specific sericulture documents are collected, curated, and preprocessed.

### 2️⃣ Embedding Generation

Sentence Transformer models convert textual data into **semantic vector embeddings**.

### 3️⃣ Vector Storage

The embeddings are stored in a **FAISS vector database** for efficient similarity search.

### 4️⃣ Query Processing

User queries are converted into embeddings and compared with stored vectors to retrieve the most relevant information.

### 5️⃣ Response Generation

A lightweight **LLM generates context-aware responses** using the retrieved knowledge.

---

## 🎯 Deployment Design

The chatbot is designed keeping **rural deployment conditions** in mind:

* Lightweight model architecture
* Minimal computational requirements
* Suitable for low-bandwidth internet environments
* Easy-to-use web interface for non-technical users

---

## 📈 Evaluation

System performance is evaluated using **semantic similarity metrics**, which measure how closely chatbot responses align with expert recommendations and expected advisory outputs.

---

## 🌱 Impact

By centralizing verified sericulture knowledge into an accessible AI system, this project:

* Supports **informed farming decisions**
* Reduces misinformation in agricultural advisory services
* Improves productivity in rural communities
* Promotes **sustainable sericulture practices**
* Demonstrates the value of **domain-specific AI solutions**

---

## 🛠️ Technologies Used

* Python
* Sentence Transformers
* FAISS (Vector Database)
* Lightweight Large Language Model (LLM)
