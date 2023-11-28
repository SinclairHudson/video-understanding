# video-understanding
A project for CSC2508 advanced data systems; understanding video through textual representations

---

LLMs are multimodal and given images you can ask them to describe them in detail. This opens new
directions. Traditionally the vision community has developed video summarization algorithms (image
based) and video query answering techniques (image based as well). Can we perform the same tasks
with LLMs?
The idea is simple: get a large video, process it frame by frame via an LLM and get the textual
description of each frame (ask the LLM to describe the objects, their type, their relationships etc). Now
you have a large collection of documents (frame description and frame identifier). You can index them
and answer queries. You can summarise the document and identify corresponding frames to stitch a
video summary visually. Frame changes slowly so there is a lot of redundancy across frames which you
should filter out.

---


### Initial Proposal, Oct 20 2023-ish

I would like to explore the last task: Video Query Processing with Text.
Specifically, I would like to use multi-modal LLMs to generate deep textual descriptions of the contents of the video, including objects in the scene as well as the actions that those objects may or may not perform. This project stands out to me because I am interested in video understanding and AI-powered video editing. I also have a background in computer vision, and am comfortable working with image and video data.

As the prompt says, traditionally video is processed frame by frame, but I would also like to explore prompting the LLM to analyse multiple sequential frames simultaneously, to understand the action in those frames. Can an LLM identify actions that occur between frames? For example, giving the LLM frames of before and after a basketball player shoots a shot, the LLM might be able to identify that a player has made a shot attempt.

After analyzing a lot of relevant/unique frames from the video, I would then store these textual representations in an index, and query them using text queries.
For this information retrieval problem, I would start with using sparse representations since they're faster and easier to work with, but would like to explore ideas from the "Semantic Search I" section of the course as well, such as DPR or ColBERT. The pipeline would take a text prompt from the user and return keyframes from the video that relate or answer the prompt.
For example, given a soccer game, I would like to analyse the game frame by frame and answer questions such as "when was the home team on defense?" and "who scored a goal, and when?". 
There are also harder queries such as "how many goals were scored during the match?", that require aggregating information temporally, across multiple frame descriptions. I may explore methods to handle such queries, time permitting.

The applications I have in mind for such technology would be closed captioning, sports highlights and montages, and also question-answering for video. Automatically generating event timestamps would also be very helpful for video editing (even if the accuracy isn't excellent).

I anticipate that one of the difficulties with the project will be finding a performant, free model to use for experimentation.
That being said, OpenFlamingo seems like a promising candidate: https://github.com/mlfoundations/open_flamingo.
I will also explore black-box APIs for multimodal LLMs, if they are inexpensive.

---

### TODO
use and get comfortable with LangChain; it's got retriever models, and a bunch of other simple LLM wrappers.
we gonna be running some locally!
