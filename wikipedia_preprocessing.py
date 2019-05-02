#!/usr/bin/env python
# coding: utf-8

# # Wikipedia Data exploration
# 
# This notebook explores the data of wikipedia dumps.
# 
# More info at: https://en.wikipedia.org/wiki/Wikipedia:Database_download
# 
# Data is gathered from https://dumps.wikimedia.org/enwiki/latest/
# 

# In[1]:


# Http download library
import urllib
# Filesystem library
from os import path
import os
# Compression libraries
import gzip
import bz2
import zipfile
# Regex library
import re
# NLP tookit
import nltk
# XML parsing library
import xml.etree.ElementTree
# Wikitext parser library
import mwparserfromhell as mwp
# Numpy
import numpy as np
from gensim.models import KeyedVectors
# Serialization
import ast
import pickle

nltk.download("punkt")

# ## Getting and preprocessing data
# 
# ### Downloading data
# 
# The data are .xml.gz files from wikipedia dumps.
# 
# The files are downloaded in the ./data folder in compressed format.
# 

# In[2]:


abstract_url = "http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract%d.xml.gz"
DATA_folder = path.join("Data", "Wikipedia")


def SIunit(n):
    """Return the number with SI suffix."""
    if n < 10e3:
        return n
    elif n < 10e6:
        return "%.2fk" % (n / 10e3)
    elif n < 10e9:
        return "%.2fM" % (n / 10e6)
    elif n < 10e12:
        return "%.2fG" % (n / 10e9)
    else:
        return "%.2fT" % (n / 10e12)

    
def download_file(url, file=None, progress=True, force=False):
    """Download the file at url in './data'.
    If progress is True (default True), progress will be printed.
    If force is True (default False), will download existing file.
    
    Return the file path string.
    """
    file_name = url.split('/')[-1]
    if file:
        file_path = file
    else:
        file_path = path.join(DATA_folder, file_name)
    if force or not path.isfile(file_path):
        u = urllib.request.urlopen(url)
        f = open(file_path, 'wb')

        if progress:
            file_size = int(u.getheader("Content-Length"))
            print("Downloading: %s Bytes: %s" % (file_name, SIunit(file_size)))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            if progress: 
                status = "\r%s/%s  [%3.2f%%]" % (SIunit(file_size_dl), SIunit(file_size), file_size_dl * 100. / file_size)
                print(status, end="")

        f.close()
        return file_path
    else:
        print("File %s exists: aborting download (use force=True to download anyway)." % file_name)
        return file_path


# ### Getting all links
# 
# The list of available dumps can be found at https://dumps.wikimedia.org/enwiki/latest/.

# In[3]:


dump_index_url = "https://dumps.wikimedia.org/enwiki/latest/"

def get_all_dumps(force=False):
    index_file = path.join(DATA_folder, "dump_index.html")
    is_abstract = re.compile("^enwiki-latest-abstract\d+\.xml\.gz$")
    is_page = re.compile("^enwiki-latest-pages-articles-multistream\d+\.xml-p\d+p\d+\.bz2$")
    data = {
        "abstracts": [],
        "pages": [],
        "all_abstracts": "enwiki-latest-abstract.xml.gz",
        "all_pages": "enwiki-latest-pages-articles-multistream.xml.bz2"
    }
    index_file = download_file(dump_index_url, index_file, progress=False, force=force)
    with open(index_file, 'r') as f:
        lines = f.read()
    for l in re.finditer('href="([^"]*)"', lines):
        if is_abstract.match(l.group(1)):
            data["abstracts"].append(l.group(1))
        elif is_page.match(l.group(1)):
            data["pages"].append(l.group(1))
    data
    return data
                


# In[4]:

if __name__ == "__main__":
    links = get_all_dumps()
    print("Number of abstract dumps: %d" % len(links["abstracts"]))
    print("Number of page dumps: %d" % len(links["pages"]))


# ## Loading and parsing file
# 
# To save disk space, the files are uncompressed on the fly using the gzip package.
# 
# The files are parsed using the xml package.

# In[5]:


def loadXML(filepath):
    """Return the xml python object from XML file.
    If the file is compressed ('.xml.gz' or '.xml.bz2' extension), will be uncompress on the fly.
    """
    extension = filepath.split(".")[-1]
    if extension == "gz":
        with gzip.open(filepath, 'r') as f:
            return xml.etree.ElementTree.parse(f).getroot()
    elif extension == "bz2":
        with bz2.open(filepath, 'r') as f:
            return xml.etree.ElementTree.parse(f).getroot()
    else:
        with open(filepath, 'r') as f:
            return xml.etree.ElementTree.parse(f).getroot()
    print("Couldn't open file %s" % filepath)
    return None



# ### Abstract file parsing
# 
# 
# Example of abstract file
# ```
# <feed>
#     <doc>
#         <title>Wikipedia: Blepharomastix ineffectalis</title>
#         <url>https://en.wikipedia.org/wiki/Blepharomastix_ineffectalis</url>
#         <abstract>Blepharomastix ineffectalis is a moth in the Crambidae family. It was described by Francis Walker in 1862.</abstract>
#     </doc>
#     <doc>
#         <title>Wikipedia: Blepharomastix costalis</title>
#         <url>https://en.wikipedia.org/wiki/Blepharomastix_costalis</url>
#         <abstract>Blepharomastix costalis is a moth in the family Crambidae. It was described by Francis Walker in 1866.</abstract>
#         <links>
#             <sublink linktype="nav"><anchor>References</anchor><link>https://en.wikipedia.org/wiki/Blepharomastix_costalis#References</link></sublink>
#         </links>
#     </doc>
# </feed>
# ```
# 
# The parsing function extracts:
# - the title (removing the leading "Wikipedia: ")
# - the abstract
# - an unique id from the url ("https://en.wikipedia.org/wiki/Blepharomastix_ineffectalis" becomes id "en/Blepharomastix_ineffectalis")
# 
# It returns a array of dictionnaries with keys: "id", "name" and "abs".

# In[6]:


def parse_abstract(xmlRoot):
    """Parse the xml abstract file."""
    # This regex extract the language and article name of the page
    extract_id = re.compile("https?://(\w+)\.wikipedia\.org/wiki/(.*)")
    extract_title = re.compile("Wikipedia: (.*)")
    data = []
    for p in xmlRoot.iter("doc"):
        page_name_raw = p.find("title").text
        page_name = extract_title.match(page_name_raw)
        page_url = p.find("url").text
        page_id = extract_id.match(page_url)
        abstract = p.find("abstract").text
        if page_id and page_name and abstract:
            data.append({
                "id": page_id.group(1) + "/" + page_id.group(2),
                "name": page_name.group(1),
                "abs": abstract
            })
    return data


# ### Page file parsing
# 
# Example of page xml file.
# ```
# <mediawiki [...]>
#     <siteinfo>
#         [...]
#     </siteinfo>
#     <page>
#         <title>Konica Minolta Cup</title>
#         <ns>0</ns>
#         <id>7697605</id>
#         <revision>
#             <id>380827672</id>
#             <parentid>377074437</parentid>
#             <timestamp>2010-08-25T01:11:11Z</timestamp>
#             <contributor>
#                 <username>Surge79uwf</username>
#                 <id>265372</id>
#             </contributor>
#             <model>wikitext</model>
#             <format>text/x-wiki</format>
#             <text xml:space="preserve">'''Konica Minolta Cup''' may refer to
#                 * [[Japan LPGA Championship]] Konica Minolta Cup, was a golf competition
#                 * [[WRU Challenge Cup]], a Welsh rugby union competition
# 
#                 '''Konica Cup''' (before the Minolta merger) may refer to
#                 * [[Konica Cup (football)]], a football competition
# 
#                 {{disambig}}</text>
#             <sha1>orasgh0bid09rp4x7ij4jwxt6g83nrx</sha1>
#         </revision>
#     </page>
#     <page>
#         <title>Archer (typeface)</title>
#         <ns>0</ns>
#         <id>7697611</id>
#         <revision>
#             <id>807146475</id>
#             <parentid>797421154</parentid>
#             <timestamp>2017-10-26T05:27:23Z</timestamp>
#             <contributor>
#                 <username>Blythwood</username>
#                 <id>18364158</id>
#             </contributor>
#             <minor />
#             <comment>Rm category it's in a subcat of</comment>
#             <model>wikitext</model>
#             <format>text/x-wiki</format>
#             <text xml:space="preserve"> [...] </text>
#         </revision>
#     </page>
# </mediawiki>
# ```
# 
# The page parsing function extracts:
# - the title
# - the id
# - the page content
# 
# Notes: Since the page urls are not in the page dump and the page ids are not in the abstract dump, it is not possible to link the two entities.  
# 
# TODO:
# - Remove Wikitext formating (https://en.wikipedia.org/wiki/Help:Wikitext)
# - Remove redirect pages
# - Remove disambiguation pages


# In[16]:


def parse_page(xmlRoot):
    """Parse the xml page file."""
    ns = xmlRoot.tag[:-9]
    data = []
    for p in xmlRoot.iter(ns + "page"):
        page_name = p.find(ns + "title").text
        page_id = int(p.find(ns + "id").text)
        page_wikitext = p.find(ns + "revision/" + ns + "text")
        if page_wikitext is not None:
            page_parsed = ' '.join(map(lambda x: x.value, mwp.parse(page_wikitext.text).ifilter_text()))
            data.append({
                "id": page_id,
                "name": page_name,
                "content": page_parsed
            })
    return data


# ### Preprocess data
# 
# For easy computation and low memory footprint, we precompute the wikipedia dumps.

# In[25]:


abs_precomp_filepath = path.join(DATA_folder, "precomp", "abstracts_raw.text.gz")
url_base = "https://dumps.wikimedia.org/enwiki/latest/"

def precompute_abstracts(keep_dumps=True, force=False, flush_buffer=50):
    if path.isfile(abs_precomp_filepath):
        if force:
            os.remove(abs_precomp_filepath)
        else:
            print("Abstract already precomputed: aborting (use force=True).")
            return
    dumps = get_all_dumps(force)
    # Open a file with on the fly compression
    with gzip.open(abs_precomp_filepath, "w") as f:
        # For every abstract dump
        for fp in dumps["abstracts"]:
            fp = download_file(url_base + fp)
            print("Processing %s" % fp)
            # Load and process .xml.bz
            e = loadXML(fp)
            abstracts = parse_abstract(e)
            print("Adding %s abstracts" % SIunit(len(abstracts)))
            # Add each abstract to the file
            # repr is used instead of csv or xml to allow simple line per line reading
            i = 0
            for a in abstracts:
                i += 1
                f.write((repr(a) + "\n").encode("utf-8"))
                if i > flush_buffer:
                    f.flush()
                    i = 0
            if not keep_dumps:
                os.remove(fp)
    return


# In[17]:


class Abstracts:
    def __init__(self):
        if not path.isfile(abs_precomp_filepath):
            raise Exception("Abstract not precomputed")
        self.file = gzip.open(abs_precomp_filepath, "r")
    
    def __iter__(self):
        return self

    def __next__(self):
        line = self.file.readline()
        if line:
            return ast.literal_eval(line.decode("utf-8"))
        else:
            return StopIteration


# In[ ]:

if __name__ == "__main__":
    print("Precomputing abstracts")
    precompute_abstracts()


# In[26]:


page_precomp_filepath = path.join(DATA_folder, "precomp", "pages_raw.text.gz")
url_base = "https://dumps.wikimedia.org/enwiki/latest/"

def precompute_pages(keep_dumps=True, force=False, flush_buffer=50, max_dumps=-1):
    if path.isfile(page_precomp_filepath):
        if force:
            os.remove(page_precomp_filepath)
        else:
            print("Pages already precomputed: aborting (use force=True).")
            return
    dumps = get_all_dumps(force)
    # Open a file with on the fly compression
    with gzip.open(page_precomp_filepath, "w") as f:
        # For every page dump
        i = 0
        for fp in dumps["pages"]:
            i += 1
            if max_dumps > 0 and i > max_dumps:
                break
            fp = download_file(url_base + fp)
            print("Processing %s" % fp)
            # Load and process .xml.bz
            e = loadXML(fp)
            pages = parse_page(e)
            print("Adding %s pages" % SIunit(len(pages)))
            # Add each page to the file
            # repr is used instead of csv or xml to allow simple line per line reading
            i = 0
            for a in pages:
                i += 1
                f.write((repr(a) + "\n").encode("utf-8"))
                if i > flush_buffer:
                    f.flush()
                    i = 0
            if not keep_dumps:
                os.remove(fp)
    return


# In[ ]:


class Pages:
    def __init__(self):
        if not path.isfile(page_precomp_filepath):
            raise Exception("Pages not precomputed")
        self.file = gzip.open(page_precomp_filepath, "r")
    
    def __iter__(self):
        return self

    def __next__(self):
        line = self.file.readline()
        if line:
            return ast.literal_eval(line.decode("utf-8"))
        else:
            return StopIteration


# In[ ]:

if __name__ == "__main__":
    print("Precomputing pages")
    precompute_pages()


# In[ ]:


# https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
word2vec_pickle_path = path.join(DATA_folder, "wiki-news-300d-1M.vec.pkl")
def get_word2vec():
    print("Loading Word2vec model.")
    if path.isfile(word2vec_pickle_path):
        word2vec = pickle.load(open(word2vec_pickle_path, "rb"))
        print("Loaded from pickle")
    else:
        word2vec_path = path.join(DATA_folder, "wiki-news-300d-1M.vec")
        word2vec = KeyedVectors.load_word2vec_format(word2vec_path)
        print("Loaded")
        print("Pickling word2vec")
        pickle.dump(word2vec, open(word2vec_pickle_path, "wb"))
        print("Done")
    return word2vec

if __name__ == "__main__":
    word2vec = get_word2vec()

# print("Loading word2vec")
# word2vec = {}
# with zipfile.ZipFile(word2vec_path, "r") as zfile:
#     with zfile.open(zfile.namelist()[0], "r") as file:
#         n, d = map(int, file.readline().split())
#         i = 0
#         for line in file:
#             print("\rWord %s/%s" % (SIunit(i), SIunit(n)), end="")
#             i += 1
#             tokens = line.decode("utf-8").rstrip().split(' ')
#             word2vec[tokens[0]] = map(float, tokens[1:])
# print("Word2vec loaded.       ")        


from nltk.tokenize import word_tokenize
abs_vec_path = path.join(DATA_folder, "precomp", "abstracts_vectorize.vec.gz")

def compute_single_vector_abstracts():
    if not path.isfile(abs_vec_path):
        abstracts = Abstracts()
        print("Calculated mean for abstracts")
        with gzip.open(abs_vec_path, "w") as file:
            vector = np.zeros(300, dtype=np.float64)
            i = 0
            for a in abstracts:
                print("\rComputed %s abstracts" % SIunit(i), end="")
                i += 1
                if type(a["abs"]) != type(" "):
                    print(a["abs"])
                tokens = word_tokenize(a["abs"])
                vector[:] = 0
                norm = 0
                for t in tokens:
                    try:
                        vector += word2vec.get_vector(t)
                        norm += 1
                    except KeyError:
                        pass
                if norm != 0:
                    file.write((repr({
                        "id": a["id"],
                        "name": a["name"],
                        "vector": (vector / norm).tolist()
                    }) + "\n").encode("utf-8"))
    else:
        print("Abstract vectors already computed: aborting.")


if __name__ == "__main__":
    compute_single_vector_abstracts()


class VecAbstracts:
    def __init__(self):
        if not path.isfile(abs_vec_path):
            raise Exception("Abstract vec not precomputed")
        self.file = gzip.open(abs_vec_path, "r")

    def __iter__(self):
        return self

    def __next__(self):
        line = self.file.readline()
        if line:
            try:
                return ast.literal_eval(line.decode("utf-8"))
            except Exception as e:
                print("Error Loading VecAbstract item:", e)
                return self.__next__()
        else:
            return StopIteration

if __name__ == "__main__":
    abstract_keyedvec_path = path.join(DATA_folder, "precomp", "abs_keyed_vec.pkl")
    keyedvec = KeyedVectors(300)
    vecAbs = VecAbstracts()
    i = 0
    for va in vecAbs:
        print("\r%d abstract keyed vector added." % i, end="")
        i += 1
        keyedvec.add([str(va["id"])], [va["vector"]])
    pickle.dump(keyedvec, open(abstract_keyedvec_path, "rb"))

pages_vec_path = path.join(DATA_folder, "precomp", "pages_vectorize.vec.gz")
def vectorize_pages(pages):
    if not path.isfile(pages_vec_path):
        print("Calculating vectorized pages")
        with gzip.open(abs_vec_path, "w") as file:
            i = 0
            for p in pages:
                doc = []
                print("\rComputed %s pages" % SIunit(i), end="")
                i += 1
                if type(p["content"]) != type(" "):
                    print(p["content"])
                tokens = word_tokenize(p["content"])
                for t in tokens:
                    try:
                        doc.append(word2vec.get_vector(t))
                    except KeyError:
                        pass
                if doc:
                    c = StringIO()
                    np.savetxt(c, np.array(doc).reshape((-1,1)))
                    if len(c) > 0:
                        print("Error, multiline c")
                    file.write((repr({
                        "id": p["id"],
                        "name": p["name"],
                        "content": c.readlines()[0]
                    }) + "\n").encode("utf-8"))
    else:
        print("Pages vectors already computed: aborting.")


class VecPages:
    def __init__(self):
        if not path.isfile(pages_vec_path):
            raise Exception("Pages vec not precomputed")
        self.file = gzip.open(pages_vec_path, "r")

    def __iter__(self):
        return self

    def __next__(self):
        line = self.file.readline()
        if line:
            try:
                p = ast.literal_eval(line.decode("utf-8"))
                c = StringIO(p["content"])
                p["content"] = np.loadtxt(c).reshape((-1, 300))
                return p
            except Exception as e:
                print("Error Loading VecPages item:", e)
                return self.__next__()
        else:
            return StopIteration
