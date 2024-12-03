import numpy as np
import matplotlib.pyplot as plt
from time import sleep


def draw_two_faces():
    w_face_blk_bg = draw_white_face()
    b_face_white_bg = 1 - w_face_blk_bg

    return w_face_blk_bg, b_face_white_bg


def draw_white_face():
    """
    Draws white face on black background
    :return: numpy matrix
    """
    w_face_blk_bg = np.zeros([14, 14])
    # Top eyes
    w_face_blk_bg[3, 2:6] = 1
    w_face_blk_bg[3, 8:12] = 1
    # Bottom eyes
    w_face_blk_bg[4, 3:5] = 1
    w_face_blk_bg[4, 9:11] = 1
    # Narrow nose
    w_face_blk_bg[6:9, 6:8] = 1
    # Wide nose
    w_face_blk_bg[9, 5:9] = 1
    # Cheeks
    w_face_blk_bg[10:12, 3] = 1
    w_face_blk_bg[10:12, 10] = 1
    # Mouth
    w_face_blk_bg[12, 3:11] = 1
    return w_face_blk_bg


def display_faces(faces, graph_title="Face Comparison", counter=None):
    """
    Displays two faces side by side with a main graph title.

    :param counter: attempts counter, if relevant
    :param faces: A tuple of two numpy arrays representing the faces.
    :param graph_title: A string representing the main title of the graph.
    """
    captions = ["Test Face", "Template Face"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))

    title_color = "black"

    status = graph_title
    if status == "access permitted":
        title_color = "green"
    elif status == "access denied":
        title_color = "red"

    fig.suptitle(graph_title, fontsize=16, y=0.95, color=title_color)

    for i in range(2):
        axes[i].imshow(faces[i], cmap='gray')
        axes[i].set_title(captions[i])
        axes[i].axis('off')

    if counter is not None:
        fig.text(0.5, 0.01, f"Attempts: {counter}", ha='center', fontsize=12)


    plt.tight_layout()
    plt.show()


def calc_resemblance_factor(A, B):
    """
    Calculates the resemblance factor between two matrices using the Frobenius inner product.

    The resemblance factor, p, is a measure of similarity between matrices A and B:
    - p = 1 indicates the matrices are identical.
    - p = 0 indicates the matrices are completely different.
    - p = 0.5 indicates that about half of the values are equal.

    :param A: numpy ndarray
        First input matrix.
    :param B: numpy ndarray
        Second input matrix.
    :return: float
        Resemblance factor, p, where 0 <= p <= 1.
    """

    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same dimensions.")

    frobenius_inner = np.sum(A * np.conj(B))

    norm_A = np.linalg.norm(A, "fro")
    norm_B = np.linalg.norm(B, "fro")

    return np.abs(frobenius_inner) / (norm_A * norm_B)


def check_access(template_face, test_face, p_threshold):
    """
    Compares two faces in order to grant access
    :param template_face: the allowed face
    :param test_face: the face in question
    :param p_threshold: the allowed resemblance factor (0.58, 0.9, whatever it may be)
    :return: "access permitted" or "access denied"
    """
    p = calc_resemblance_factor(template_face, test_face)

    return "access permitted" if p > p_threshold else "access denied"


def face_factory():
    """
    Produces a binary matrix, "face", with edges always set to 1.
    :return: binary matrix, numpy array
    """
    matrix = np.random.randint(2, size=(14, 14))

    # Set the frame to 0 (black)
    matrix[0, :] = 0
    matrix[-1, :] = 0
    matrix[:, 0] = 0
    matrix[:, -1] = 0

    return matrix

def hostile_takeover():
    """
    Simulates a hostile takeover - a brute force attack
    """
    template_face = draw_white_face()
    open_gate = False
    attempts = 0
    while not open_gate:
        hostile_face = face_factory()
        access_status = check_access(template_face, hostile_face, 0.59)
        display_faces((hostile_face, template_face), access_status, attempts)
        sleep(0) # Add a little delay
        attempts+=1

        open_gate = (access_status == "access permitted")

def main():
    # faces = draw_two_faces()
    # print("Resemblance Factor:", calc_resemblance_factor(faces[0], faces[1]))
    # display_faces(faces)

    hostile_takeover()

if __name__ == '__main__':
    main()

