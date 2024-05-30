    import requests
    from bs4 import BeautifulSoup
    import nltk
    nltk.download("punkt")
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    import tkinter as tk
    from collections import defaultdict
    from tkinter import Scrollbar, Listbox, END, RIGHT, Y
    
    def create_index():
        return {}
    
    def extract_publications_data(url):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        all_publications = soup.find_all("div", class_="rendering_researchoutput_portal-short")
    
        publications_data = []
    
        for publication in all_publications:
            idx = publication.index
            title_element = publication.find("h3", class_="title")
            title_text = title_element.a.text.strip() if title_element and     title_element.a else "Title not available"
            publication_link = title_element.a["href"] if title_element and title_element.a else "Link not available"
    
            authors = publication.find_all("a", class_="link person")
            author_text = ", ".join(author.text.strip() for author in authors) if authors else "Author not available"
            
            author_links = [author["href"] for author in publication.find_all("a", class_="link person")]
            author_links_text = ", ".join(author_links) if author_links else "Author links not available"
    
            publication_date_element = publication.find("span", class_="date")
            if publication_date_element:
                publication_date_text = publication_date_element.text.strip()
            else:
                publication_date_text = "Publication Date not available"
    
            publications_data.append({
                "Publication Title": title_text,
                "Publication Date": publication_date_text,
                "Publication Link": publication_link,
                "Authors": author_text,
                "Author Links": author_links_text,
            })
    
        return publications_data
    
    def crawl_and_extract(base_url, num_pages):
        publications_data = []
        page_num = 0
        
        while page_num <= num_pages:
            url = f"{base_url}?page={page_num}"
            print("Fetching data...")
            page_data = extract_publications_data(url)
            publications_data.extend(page_data)
            page_num += 1
        
        return publications_data
    
    
    def preprocess_text(text):
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        terms = word_tokenize(text.lower())
        filtered_text = [stemmer.stem(term) for term in terms if term not in stop_words]
        return " ".join(filtered_text)
    
    def build_inverted_index(publications_data):
        term_to_documents = defaultdict(list)
    
        n = 1
        while n <= len(publications_data):
            publication = publications_data[n - 1]
            text_content = f"{publication['Publication Title']} {publication['Authors']} {publication['Publication Date']}"
            filtered_text = preprocess_text(text_content)
    
            for term in filtered_text.split():
                term_to_documents[term].append(n)
    
            publication["Filtered Text"] = filtered_text
    
            n += 1
    
        return term_to_documents
    
    
    def rank_matched_documents(query_terms, term_to_documents):
        matched_docs = {}
        term_index = 0
    
        while term_index < len(query_terms):
            term = query_terms[term_index]
            if term in term_to_documents:
                docs = term_to_documents[term]
                doc_index = 0
    
                while doc_index < len(docs):
                    doc = docs[doc_index]
                    matched_docs[doc] = matched_docs.get(doc, 0) + 1
                    doc_index += 1
            
            term_index += 1
    
        ranked_docs = sorted(matched_docs.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked_docs]
    
    
    
    def search_publications(query, term_to_documents, publications_data):
        query_terms = word_tokenize(query.lower())
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        query_terms = [stemmer.stem(term) for term in query_terms if term not in stop_words]
        matched_docs = rank_matched_documents(query_terms, term_to_documents)
        matched_publications = [publications_data[idx - 1] for idx in matched_docs]
        if not matched_publications:
            print("No publications found.")
            return
        print("Found", len(matched_publications), "Matching Publications '", query, "':\n")
        i = 0
        while i < len(matched_publications):
            publication = matched_publications[i]
            print("Fetched Data:")
            for key, value in publication.items():
                if key == "Authors":
                    print(key + ":", value)
                elif key == "Author Links":
                    links = value.split(", ")
                    for link in links:
                        print(key + ":", link)
                else:
                    print(key + ":", value)
            print("-" * 40)
            i += 1
            
    if __name__ == "__main__":
        base_url = "https://pureportal.coventry.ac.uk/en/organisations/centre-global-learning/publications/"
        num_pages = 6  
        publications_data = crawl_and_extract(base_url, num_pages)
        term_to_documents = build_inverted_index(publications_data)
        while True:
            user_query = input("Search For Publications or Authors: ")
            search_publications(user_query, term_to_documents, publications_data)
    
