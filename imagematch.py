"""
-Use the trained embedding matrix to convert each images descriptor vector (shape-
) to a corresponding embedding vector (shape-
).
-Create image database that maps image ID -> image embedding vector
-Write function to query database with a caption-embedding and return the top-k images
-Write function to display a set of 
 images given their URLs.

"""

import numpy as np
import io

import requests
from PIL import Image, ImageDraw, ImageFont

def download_image(img_url: str) -> Image:
    """Fetches an image from the web.

    Parameters
    ----------
    img_url : string
        The url of the image to fetch.

    Returns
    -------
    PIL.Image
        The image."""

    response = requests.get(img_url)
    return Image.open(io.BytesIO(response.content))

def convert_descriptor_to_embed (image_ids, descriptor_vectors, embedding_matrix):
    """
    Converts each images descriptor vector into a 
    embedding vector, and then stores it inside a dictionary
    Args: image_ids: list of image IDs
          descriptor_vectors: Dictionary mapping each image_id to its descriptor vector
          embedding_matrix: trained embedding matrix
    
    Returns: image_embeddings: Dictionary mapping each image_id to its embedding vector

    """
    #empty dictionary to hold image embeddings
    image_embeddings = {}
    
    for image_id in image_ids: 
        #get the descripto vector for the specific image from the decripto vector list
        descriptor_vector= descriptor_vectors[image_id]
        # convert the descriptor vector to a embedding ny multiplying w embed matrix
        embedding_vector= descriptor_vector @ embedding_matrix
        #normalize the embedding vector, just so that no score dominates based on sheer length
        embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
        #store the vector in the dictionary, under the image ID specified
        image_embeddings[image_id] = embedding_vector
    #return the embedding dictionary
    return image_embeddings

def query_db(caption_embedding, image_embeddings, topk=4):
    """
    Finds the the top k (int value) imagges from the database
    that are most simuiliar to the input query, which would be ranked 
    based on similarity of the dot product

    args: 
    caption_embedding- the embedding vector to rep 
    the caption

    image_embeddings- the map from the image id to 
    embeddding vector found from last function

    top_k- the number of images to be returned, 
    starting with 4 suibject to change tho

    Returns:
        a list of tuplex that would show each image_id and 
        score of similarity in  order

    """
    #empty list of scores
    scores=[]
    for image_id, embedding_vector in image_embeddings.items():
        #find the similiarity between the quesry and embedding vector
        similarity= caption_embedding@ embedding_vector
        #appened each score into the scores list
        scores.append(image_id, similarity)
    #sort all images so that the highest show, and only then goes up till k
    scores.sort(key=lambda x: x[1])
    #reverses the scores, so that biggest is fitst not smallest
    scores.reverse()
    #return the top k images
    return scores[:topk]
    



def display_images(data, image_ids):
      """
      Takes the image IDs of the top-k similar scores and displays those images
      
      args:

      data (CocoData) : a CocoData object to access the urls from the josn files
      
      image_ids (Tuple) : a tuple of image_ids in descending order of score similarity
            
      returns:

      None (Displays the images)
      """
      images = []
      captions = []
      font = ImageFont.truetype("FreeMono.ttf", 24)
      for k, (image_id, score) in enumerate(image_ids):
          url = data.get_url_from_image_id(image_id)
          images.append(download_image(url))
          captions.append(data.get_captions_from_image_id(image_id))

      for index,image in enumerate(images):
            draw = ImageDraw.draw(image)
            draw.text((10,10), captions[index], font = font, fill = (255,0,0))
            image.show()

      



      
      


