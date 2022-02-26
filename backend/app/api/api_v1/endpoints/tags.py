from app.core.classifier import load_classifier, load_vectorizer
from app.models.tags import Sentance, TaggedSentence
from fastapi import APIRouter

router = APIRouter()


@router.post("/", response_model=TaggedSentence)
async def tag(sentence: Sentance) -> TaggedSentence:
    clf = load_classifier()
    vectorizer = load_vectorizer()
    tags = list(clf.predict(vectorizer.transform([sentence.sentence])))
    return TaggedSentence(sentence=sentence.sentence, tags=tags)
