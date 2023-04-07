'''
[`기존`]
새 질문 -> 독립형 질문 생성 -> 문서 내용 탐색 지시 -> 히스토리 기록

[standalone, 독립형 질문 생성 프롬프트]
{chat_history} = 히스토리, {question} = 새 질문
'Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question.
Provide answers in question language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:'


[문서 내용 탐색 지시 프롬프트]
{question} = 독립형 질문, {context} = 문서
'You are an AI assistant providing helpful advice.
You are given the following extracted parts of a long document and a question.
Provide a conversational answer based on the context provided.
You should only provide hyperlinks that reference the context below.
Do NOT make up hyperlinks.
If you can't find the answer in the context below,
just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not related to the context,
politely respond that you are tuned to only answer questions that are related to the context.
Question: {question}
=========
{context}
=========
Answer in Markdown:'


[히스토리 생성]
history = ['유저 질문', 'GPT 대답']
'''

'''
[`수정`]
새 질문 -> 독립형 질문 생성 -> 문서 내용 탐색 지시 -> 히스토리 기록

[standalone, 독립형 질문 생성 프롬프트]
{chat_history} = 히스토리, {question} = 새 질문
'I give you a Chat History of someone and me chatting.
New Question is the next question following the chat history.
Please create a new "Standalone Question" that combines chat history and new questions.
Please reply in korean.
Chat History: {chat_history}
New Question: {question}'


[문서 내용 탐색 지시 프롬프트]
{question} = 독립형 질문, {context} = 문서
'I give you my question and the document.
You should only reply that reference the document below.
Generation of unreferenced answers is prohibited.
If you can't find the answer in the document below, just say "Hmm, I'm not sure."
Don't make up hyperlinks.
Please reply in korean.
Question: {question}
Document: {context}'


[히스토리 생성]
history = [f"me: {question}\n”, f”someone: {answer}\n”]
'''

'''
[2차 수정]

[standalone 프롬프트]
삭제
New Question: I give you chat history and question.\n{chat_history}\n{question}


[문서 내용 탐색 프롬프트]
I give you chat history and question and document.
Please reply in language used in the question.

Question: {question}
Document: {context}

Please summarize the document into 500~550 characters.


'''

'''
간단한 작업에 대해서는 더 빠르게 수행가능 함.
(키워드 요약 지시)
기존(12.2초) : 내용 동일함. 
수정(8.1초) : 내용 동일함.

요약 작업에 대하여 맥락을 더 잘 이해 했으며,
생성된 답변이 더 자연스러운 편임.
(문서 전체 요약)
기존(31.8초) : 이 문서는 미래를 예측하는 다양한 단서 중 하나인 '사람'에 초점을 맞추고 있습니다. 각 세대의 특성을 규정하는 계기 중 가장 큰 영향력을 미치는 사건은 코로나19로, 모든 세대가 영향을 받았지만 특히 어린 세대가 더 크게 영향을 받았습니다. 코로나19로 인해 비대면 상황이 뉴노멀이 되었고, 어린 시절에 포스트 팬데믹 시대를 겪은 세대는 알파 세대와 후기 Z세대로 볼 수 있습니다. 알파 세대는 2010년에서 2024년에 태어난 아이들로, Z세대 다음 세대이고 밀레니얼의 자녀입니다.
수정(35.2초) : 본 문서는 미래를 예측하는 다양한 단서 중 하나인 '사람'에 초점을 맞추고 있습니다. 각 세대의 특성을 규정하는 계기 중 하나로 코로나19가 큰 영향력을 미치고 있다고 설명하며, 이로 인해 모든 세대가 영향을 받았지만 특히 어린 세대가 더 크게 영향을 받았다고 합니다. 어린 시절에 포스트 팬데믹 시대를 겪은 세대는 알파 세대와 후기 Z세대로 볼 수 있으며, 이들은 비대면 상황으로 인해 정상적인 사회화 과정을 거치지 못하고 있습니다.
'''

'''
[`100단어 내외 요약 실험`]
목차별 요약 / 키워드 (+핵심 아이디어)

질문: 
다음 3개의 문서를 요약하는 리포트를 100단어 내외로 작성해줘.
경제 연구원 말투로 부탁해.

1번 문서: 
2021년 임금근로자 대출 평균액이 1년 전보다 7% 증가한 5202만원으로 집계됐다. 20대 근로자 대출 증가세가 눈에 띄게 높아졌으며, 40대 근로자가 가장 많은 7638만원의 대출액을 가지고 있었다. 소득이 높을수록 대출액도 많아지는 추세이며, 대기업 근로자의 평균대출액은 중소기업 근로자의 1.9배를 기록했다. 다만, 금융당국의 가계부채 관리 강화로 대출 증가율은 둔화했다. 대출 연체율은 전 연령대에서 가장 높은 20대 임금근로자가 1.54%로 나타났다.
2번 문서: 
3월 중순 실리콘 밸리 은행의 파산은 유동성, 체계적, 실질적인 경기 순환 위기의 형태로 계속해서 전개되는 미국 지역 은행 위기를 촉발했습니다. 바이든 행정부는 도드-프랭크 법을 위기 대응의 근거로 삼아 위기가 시스템 위기로 번지는 것을 막기 위해 최선을 다하고 있습니다. 예금자 보호를 보장하고 책임져야 할 금융기관과 투자자에게 책임을 물으면서 유동성 위기를 해결하는 것이 최우선 과제입니다. 하지만, 신용 위기의 위험과 비정상적인 새로운 위험은 그 문제를 해결하기 어렵게 만듭니다. 게다가, 중국의 채권 매각은 미국 금융 시스템에 추가적인 압박을 가하고 있습니다.
3번 문서: 
반도체 산업에 대한 엇갈린 전망에도 불구하고 국내 투자자들은 반도체 테마의 상장지수펀드(ETF)에 대한 사랑을 잃을 기미를 보이지 않고 있습니다. 인공지능 개발은 반도체 시장의 장기적인 성장에 기여할 것으로 여겨지는데, 이 때문에 점점 더 많은 투자자들이 이런 유형의 투자로 눈을 돌리고 있습니다. 시가총액 상위 10개 반도체 기업에 투자하는 TIGER Fn 반도체 TOP10 ETF는 올해 개인투자자들이 128억원어치의 주식을 사들이며 순자산 1조원을 돌파했습니다. 다른 반도체 ETF도 상당한 유입을 경험해 대부분 올해 국내주식 ETF의 평균 수익률(11.64%)을 앞질렀습니다.

요약 리포트 내용: 
2021년 대출 증가율이 둔화되었지만, 소득이 높아질수록 대출액도 높아지는 추세를 보이며, 20대 대출 증가율이 눈에 띄게 높아졌다. 대출 연체율은 20대가 가장 높았다. 미국 지역 은행 위기가 촉발된 3월 중순 실리콘 밸리 은행 파산 사건은 유동성, 체계적, 실질적인 경기 순환 위기로 계속해서 전개되고 있으며, 바이든 행정부는 위기가 시스템 위기로 번지는 것을 막기 위해 최선을 다하고 있다. 반면에 국내 투자자들은 반도체 테마 ETF에 대한 관심을 계속 유지하고 있으며, TIGER Fn 반도체 TOP10 ETF는 개인 투자자들이 순자산 1조원을 돌파하는 등 상당한 인기를 끌고 있다.

{num_doc} : 문서 갯수 / {lang_type} : 새 보고서의 말투 / {doc1} / {doc2}
Please write a report summarizing the following {num_doc} documents in about 100 words.
Please speak like {lang_type}.

document 1:
{doc1}

document 2:
{doc2}
'''

'''
5W1H를 기반으로 자세한 사항을 명령
Executive Summary + Details
일단 한 문서의 요약은 500자가 넘어야 함.
500, 750, 1000으로 실험.

<Executive Summary 영역>
- 타이틀 : 30자 내
- 써머리 : 300자 내로 보고서 전체를 요약한 글
- 본문 : 
Bullet Point로 구성
Bullet Point는 대표 키워드와 간단한 설명이 있음.
대표 키워드는 20자 내, 간단한 설명은 100자 내

<Details 영역>
Bullet Point에 대한 details
(타이틀-본문) 반복되는 구조
- 타이틀 : Bullet Point를 기반으로 하나씩 제시되며 30자 내
- 본문 : 
Bullet Point 마다 자세히 설명, 300자 내

[전체 프로세스]
크롤링 -> 전처리 -> Gen 레포트(1000자)(=Details) -> Summary

[전처리 프로세스]
2500자씩 썰기 -> 500자 요약 -> 취합 -> 다시 2500자 썰기 -> 500자 요약 -> 취합
500자로 최종 문서 만들기


[Preprocess 프롬프트](완료)
I'll give you the document, remember this.
Please summarize the document into 500-550 Korean characters. Please reply in Korean.


[Gen 프롬프트](완료)
I give you three documents.
I want to create a 1700 Korean characters new content that combines three documents.
Choose 5 key topics and create 5 new contents that details.
Please reply in Korean.
Document1: {document_1}
Document2: {document_2}
Document3: {document_3}


[Summary 프롬프트]
I give you the new structure and the document.
Please summarize the document according to the new structure.
Please reply in Korean.
New Structure: 
1. Title
2. 300-character summary
3-1. key topic
3-2. Keywords and 100-character descriptions of key topics
Document: {context}


[목차 대체 프롬프트](완료)
Please extract 5 key topics from the document in list type.
Please reply in Korean.
'''

