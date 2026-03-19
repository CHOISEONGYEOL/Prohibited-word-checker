# =========================
# Policy Rules Database
# =========================
# 전면 수정: 단어경계(\b) 추가, 패턴 충돌 해결, 우선순위 재정렬
# 각 규칙: pattern, label, replacement, confidence, source, aliases, delete_with_particle(optional)

RULES = [
    # ===========================
    # 상호명, 플랫폼명
    # ===========================
    {"pattern": r"(?:NAVER|Naver|naver|네이버|Daum|daum|다음|Kakao|kakao|카카오|\bGoogle\b(?!\s*(?:Docs|Classroom|TV|Maps|Drive|Slides|Forms|문서|클래스룸|티비|맵|드라이브|슬라이드|설문))|google(?!\s*(?:docs|classroom|tv|maps|drive|slides|forms))|구글(?!\s*(?:독스|클래스룸|티비|맵|드라이브|슬라이드|설문)))", "label": "상호명", "replacement": "포털사이트", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "Google(구글), NAVER(네이버), Daum(다음), Kakao(카카오) 등 → 포털사이트"}, "aliases": ["googel", "gooogle", "구글검색", "네이버검색", "다음검색", "카카오검색", "네이벌", "구굴", "네이바"]},
    {"pattern": r"(?:Google\s?Classroom|구글\s?클래스룸|EBS\s?온라인클래스|classting|클래스팅)", "label": "상호명", "replacement": "학습 플랫폼", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "Google Classroom(구글 클래스룸), EBS 온라인클래스 등 → 학습 플랫폼"}, "aliases": ["gclassroom", "구클", "클래스팅앱", "이비에스 온라인클래스"]},
    {"pattern": r"(?:TikTok|틱톡)", "label": "상호명", "replacement": "엔터테인먼트 플랫폼", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "TikTok(틱톡) 등 → 엔터테인먼트 플랫폼"}, "aliases": ["틱톡영상", "tiktoc", "틱톡스", "틱톡릴스"]},
    {"pattern": r"(?:YouTube|유튜브|TVING|티빙|watcha|왓챠|netflix|넷플릭스|wavve|웨이브|disney\s?plus|디즈니\+?|디즈니플러스|\bOTT\b)", "label": "상호명", "replacement": "동영상 플랫폼", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "YouTube(유튜브), TVING(티빙) ... OTT 등 → 동영상 플랫폼"}, "aliases": ["yutube", "you tube", "유튭", "유툽", "유투브", "넷플", "왓챠플레이", "유튜브쇼츠", "쇼츠", "shorts", "유튜브영상"]},
    {"pattern": r"(?:YouTuber|유튜버)", "label": "직업명", "replacement": "동영상 크리에이터", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "YouTuber(유튜버) 등 → 동영상 크리에이터, 동영상 제공자"}, "aliases": ["유튜브러", "youtuber"]},
    {"pattern": r"(?:KakaoTalk|카카오톡|카톡|\bLINE\b|(?<![가-힣])라인(?![가-힣])|Instagram|인스타그램|Twitter|트위터|\bMeta\b|메타|Facebook|페이스북)", "label": "상호명", "replacement": "소셜 네트워크 서비스", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "KakaoTalk, Instagram, Facebook 등 → 메신저, 소셜네트워크서비스"}, "aliases": ["kakaotalk", "kkt", "카톡방", "인스타", "insta", "페북", "x(트위터)", "인스타릴스", "릴스", "인스타그램릴스"]},
    {"pattern": r"(?:Chat\s?GPT|챗\s?GPT|챗지피티|wrtn|뤼튼|bing\s?Chat|빙챗|\bBard\b|바드|하이퍼클로바X|HyperClova\s?X|Gemini|제미나이|\bClaude\b|클로드)", "label": "상호명", "replacement": "생성형 인공지능", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "Chat GPT(챗지피티), wrtn(뤼튼) ... 등 → 대화형 인공지능, 생성형 인공지능"}, "aliases": ["chatgpt", "챗쥐피티", "gpt챗", "빙챗봇", "하클x", "뤼튼ai", "지피티", "GPT", "쳇지피티", "gemini ai", "챗쥐피디", "쳇gpt"]},
    {"pattern": r"(?:Canva|캔바|miricanvas|미리캔버스|mangoboard|망고보드)", "label": "상호명", "replacement": "디자인 제작 플랫폼", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "miricanvas(미리캔버스), mangoboard(망고보드), Canva(캔바) 등"}, "aliases": ["캔바앱", "미캔"]},
    {"pattern": r"(?:KineMaster|키네마스터|Premiere\s?Pro|프리미어\s?프로)", "label": "프로그램명", "replacement": "영상 편집 프로그램", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "영상 제작 프로그램, 영상 편집 프로그램"}, "aliases": ["키네", "프리미어"]},

    # ===========================
    # 개발 언어 / 개발 도구
    # ===========================
    {"pattern": r"(?:\bPython\b|파이썬|\bJava\b(?!\s*[Ss]cript)|자바(?!스크립트)|\bC\+\+|C언어|자바\s?언어|자바\s?스크립트|Java\s*Script|JavaScript|Javascript|(?<![A-Za-z])JS(?![A-Za-z]))", "label": "프로그램명", "replacement": "프로그래밍 언어", "confidence": 0.92, "source": {"doc": "대체표현", "page": 2, "quote": "프로그램명 (파이썬, C언어 등) 기재 불가"}, "aliases": ["파이선", "자바스크립", "씨언어", "javascript", "java script", "js", "파이톤", "자바언어"]},
    {"pattern": r"(?:Jupyter|주피터|Colab|코랩|PyCharm|파이참|VS\s?Code|Visual\s?Studio\s?Code|비주얼\s?스튜디오\s?코드|Anaconda|Spyder)", "label": "프로그램명", "replacement": "개발 도구", "confidence": 0.92, "source": {"doc": "대체표현", "page": 2, "quote": "특정 소프트웨어/개발환경 기재 지양, 일반화 표현 사용"}, "aliases": ["주피터", "코랩", "파이참", "vscode", "vs코드", "아나콘다", "스파이더"]},

    # ===========================
    # 오피스 / 문서 (워드 단독 사용 제거 → MS 접두 필수)
    # ===========================
    {"pattern": r"(?:MS\s?워드|MS\s?Word|Microsoft\s?Word)", "label": "프로그램명", "replacement": "문서작성 프로그램", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "hwp, MS워드 → 문서작성 프로그램"}, "aliases": ["msword", "워드파일", "엠에스워드"]},
    {"pattern": r"(?:Google\s?Docs|구글\s?문서|구글\s?독스)", "label": "프로그램명", "replacement": "온라인 문서 편집기", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "Google Docs(구글문서) 등 → 온라인 문서 편집기"}, "aliases": ["gdocs"]},

    # ===========================
    # 강연 / 이동수단 / 화상회의 (TED에 \b 추가)
    # ===========================
    {"pattern": r"(?:\bTED\b|테드)", "label": "강연명", "replacement": "온라인 강연회", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "TED(테드) 등 → 온라인 강연회"}, "aliases": ["ted 강연", "테드톡"]},
    {"pattern": r"(?:\bKTX\b|케이티엑스|\bSRT\b|에스알티)", "label": "상호명", "replacement": "고속 열차", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "KTX, SRT → 고속 열차"}, "aliases": ["케텍", "에스알티"]},
    {"pattern": r"(?:\bZoom\b|(?<![가-힣])줌(?![가-힣])|웨일온|Whale\s?ON)", "label": "상호명", "replacement": "화상 회의", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "Zoom(줌) 등 → 화상 회의"}, "aliases": ["줌미팅", "웨일온회의"]},

    # ===========================
    # 기관명 (모든 영문 약어에 \b 추가)
    # ===========================
    {"pattern": r"(?:\b(?:UN|EU|ASEAN|APEC|G7|G20|WHO|WTO|OECD|IMF|IAEA|NATO|UNESCO|UNICEF|UNEP|UNDP|UNHCR)\b|유엔|유럽연합)", "label": "기관명", "replacement": "국제기구", "confidence": 0.98, "source": {"doc": "단체명 기재", "page": 1, "quote": "교육관련기관 제외 특정 기관명 기재 불가"}, "aliases": ["유엔기구", "오이시디", "나토", "유네스코한국위원회"]},

    # ===========================
    # 논문 / 외국어 / 특수문자
    # ===========================
    {"pattern": r"소논문|연구보고서", "label": "논문 실적", "replacement": "탐구 활동", "confidence": 0.99, "source": {"doc": "논문 기재", "page": 1, "quote": "자율탐구활동 산출물 실적 기재 불가"}, "aliases": ["소논문 작성", "연구보고서를 제출"]},
    {"pattern": r"[一-龥]", "label": "외국어", "replacement": None, "confidence": 0.99, "source": {"doc": "외국어 기재", "page": 1, "quote": "한글 사용 원칙. 영문 제외 외국어 입력 불가."}, "aliases": []},
    {"pattern": r"·", "label": "특수문자", "replacement": ", ", "confidence": 0.99, "source": {"doc": "특수문자", "page": 1, "quote": "서술형 특수문자 입력 지양"}, "aliases": []},
    {"pattern": r"[※▷▶]", "label": "특수문자", "replacement": " ", "confidence": 0.99, "source": {"doc": "특수문자", "page": 1, "quote": "서술형 특수문자 입력 지양"}, "aliases": []},

    # ===========================
    # 국내 연구기관
    # ===========================
    {"pattern": r"\bKIOST\b", "label": "기관명", "replacement": "해양과학기술원", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bKIGAM\b", "label": "기관명", "replacement": "지질자원연구원", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bNOAA\b", "label": "기관명", "replacement": "미국해양대기청", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bIAU\b", "label": "기관명", "replacement": "국제천문연맹", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": []},

    # ===========================
    # 과학 전문 용어
    # ===========================
    {"pattern": r"\bGIC\b", "label": "전문 약어", "replacement": "지자기유도전류", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bEEZ\b", "label": "전문 약어", "replacement": "배타적 경제수역", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bK-VENT\b", "label": "전문 약어", "replacement": "호흡기 감염병 위험도 평가툴", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},

    # ===========================
    # 우주/항공 기관명
    # ===========================
    {"pattern": r"\bNASA\b", "label": "기관명", "replacement": "미국항공우주국", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": ["나사"]},
    {"pattern": r"\bESA\b", "label": "기관명", "replacement": "유럽우주국", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bJAXA\b", "label": "기관명", "replacement": "일본우주항공연구개발기구", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bSpaceX\b", "label": "기관명", "replacement": "민간 우주개발 기업", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "특정 기업명은 일반화 표현 사용"}, "aliases": ["스페이스엑스"]},

    # ===========================
    # 해양/과학 전문 약어
    # ===========================
    {"pattern": r"\bAUV\b", "label": "전문 약어", "replacement": "자율무인잠수정", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bROV\b", "label": "전문 약어", "replacement": "원격조종무인잠수정", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bGPS\b", "label": "전문 약어", "replacement": "위성항법장치", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bLiDAR\b", "label": "전문 약어", "replacement": "레이저 거리측정장치", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": ["라이다"]},
    {"pattern": r"\bSODAR\b", "label": "전문 약어", "replacement": "음파 탐지장치", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bSONAR\b", "label": "전문 약어", "replacement": "수중 음파 탐지기", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": ["소나"]},
    {"pattern": r"\bRADAR\b", "label": "전문 약어", "replacement": "전파 탐지기", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": ["레이더"]},

    # ===========================
    # 환경/에너지 전문 약어
    # ===========================
    {"pattern": r"\bCO2\b", "label": "전문 약어", "replacement": "이산화탄소", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "화학식은 한글명으로 대체"}, "aliases": []},
    {"pattern": r"\bPM2\.5\b", "label": "전문 약어", "replacement": "초미세먼지", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bPM10\b", "label": "전문 약어", "replacement": "미세먼지", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bLED\b", "label": "전문 약어", "replacement": "발광다이오드", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bIoT\b", "label": "전문 약어", "replacement": "사물인터넷", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bAI\b", "label": "전문 약어", "replacement": "인공지능", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bVR\b", "label": "전문 약어", "replacement": "가상현실", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bAR\b", "label": "전문 약어", "replacement": "증강현실", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bDNA\b", "label": "전문 약어", "replacement": "디옥시리보핵산", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bRNA\b", "label": "전문 약어", "replacement": "리보핵산", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bPCR\b", "label": "전문 약어", "replacement": "중합효소연쇄반응", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},

    # ===========================
    # 학술 용어(일반화)
    # ===========================
    {"pattern": r"(?:CRISPR-?Cas9|크리스퍼-?카스9?)", "label": "전문 용어", "replacement": "유전자 가위 기술", "confidence": 0.93, "source": {"doc": "학술 용어 일반화", "page": 1, "quote": "과도한 전문용어는 일반화/설명적 표현 사용 권장"}, "aliases": ["crispr", "cas9", "크리스퍼"]},

    # ===========================
    # 기재불가 공인어학시험 (모든 짧은 약어에 \b 추가)
    # ===========================
    {"pattern": r"(?:\bTOEIC\b|\bTOEFL\b|\bTEPS\b|토익|토플|탭스)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": ["토익점수", "토플점수", "탭스점수", "토익시험", "토플시험", "탭스시험"], "delete_with_particle": True},
    {"pattern": r"(?:\bHSK\b|에이치에스케이)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:\bJPT\b|\bJLPT\b|제이피티|제이엘피티)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:\bDELF\b|\bDALF\b|델프|달프)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:\bZD\b|\bTESTDAF\b|\bDSH\b|\bDSD\b|테스트다프)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:\bTORFL\b|토르플)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:\bDELE\b|델레)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:상공회의소\s?한자시험|한자능력검정|한능검|실용한자|한자급수자격검정|YBM\s?상무한검|한자급수인증시험|한자자격검정)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": ["한자시험", "한검", "한능검시험"], "delete_with_particle": True},
    {"pattern": r"(?:\bOPIC\b|\bOPIc\b|오픽)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": ["오픽시험", "오픽점수"], "delete_with_particle": True},
    {"pattern": r"(?:\bFLEX\b|플렉스)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": ["플렉스시험"], "delete_with_particle": True},
    {"pattern": r"(?:\bSNULT\b|스널트)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:\bIELTS\b|아이엘츠)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": ["아엘츠"], "delete_with_particle": True},
    {"pattern": r"(?:\bTOPIK\b|토픽|한국어능력시험)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": ["토픽시험"], "delete_with_particle": True},
    {"pattern": r"(?:\bGTQ\b|지티큐|\bITQ\b|아이티큐)", "label": "자격시험", "replacement": "", "confidence": 0.99, "source": {"doc": "자격시험 기재불가", "page": 1, "quote": "민간자격시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:\bMOS\b|모스|컴활|컴퓨터활용능력|워드프로세서|워프)", "label": "자격시험", "replacement": "", "confidence": 0.99, "source": {"doc": "자격시험 기재불가", "page": 1, "quote": "민간자격시험 성적 기재 불가"}, "aliases": ["컴활시험", "모스자격증"], "delete_with_particle": True},

    # ===========================
    # 누락된 기재 유의어
    # ===========================
    {"pattern": r"(?:Gather\s?Town|개더타운|ZEPETO|제페토|ifland|이프랜드)", "label": "상호명", "replacement": "메타버스 플랫폼", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "Gather Town(개더타운), ZEPETO(제페토) 등 → 메타버스 플랫폼"}, "aliases": ["게더타운", "제페토앱"]},
    {"pattern": r"(?:Google\s?TV|구글\s?티비)", "label": "상호명", "replacement": "동영상 플랫폼", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "Google TV(구글 티비) 등 → 동영상 플랫폼"}, "aliases": []},
    {"pattern": r"(?:Vllo|블로|Final\s?Cut\s?Pro|파이널\s?컷\s?프로)", "label": "프로그램명", "replacement": "영상 편집 프로그램", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "Vllo(블로), Final Cut Pro(파이널 컷 프로) 등 → 영상 편집 프로그램"}, "aliases": ["파컷프로", "fcpx"]},
    {"pattern": r"(?:Padlet|패들렛|ThinkerBell|띵커벨|Allo|알로)", "label": "상호명", "replacement": "온라인 협업 플랫폼", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "Padlet(패들렛), ThinkerBell(띵커벨), Allo(알로) 등 → 협업 플랫폼"}, "aliases": ["패들릿"]},
    {"pattern": r"(?:careernet|커리어넷|majormap|메이저맵)", "label": "상호명", "replacement": "진로 정보 사이트", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "careernet(커리어넷), majormap(메이저맵) 등 → 진로정보망, 진로 정보 사이트"}, "aliases": ["커리어넷검사", "메이저맵검사"]},
    {"pattern": r"(?:Holland|홀랜드)\s?검사", "label": "검사명", "replacement": "직업선호도 검사", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "Holland(홀랜드) 검사 등 → 직업선호도 검사"}, "aliases": ["홀란드검사", "홀랜드직업검사"]},
    {"pattern": r"\b(?:MBTI|엠비티아이)\b", "label": "검사명", "replacement": "성격유형 검사", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "MBTI(엠비티아이) 등 → 성격유형 검사"}, "aliases": ["mbti검사", "엠비티아이검사"]},
    {"pattern": r"\b(?:HTML|에이치티엠엘)\b", "label": "프로그램명", "replacement": "웹 페이지 제작 언어", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "HTML(에이치티엠엘) 등 → 하이퍼텍스트 마크업 언어, 웹 페이지 제작 언어"}, "aliases": ["html5", "에치티엠엘"]},
    {"pattern": r"\b(?:CSS|씨에스에스)\b", "label": "프로그램명", "replacement": "스타일 시트 언어", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "CSS(씨에스에스) 등 → 스타일 시트 언어"}, "aliases": ["css3", "씨에쎄스"]},
    {"pattern": r"(?:iPad|아이패드|Galaxy\s?Tab|갤럭시\s?탭)", "label": "상호명", "replacement": "태블릿PC", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "iPad(아이패드), Galaxy Tab(갤럭시탭) 등 → 태블릿PC"}, "aliases": ["갤탭", "아이패드프로"]},
    # Chromebook은 브라우저 규칙보다 먼저 배치 (Chrome 충돌 방지)
    {"pattern": r"(?:chrome\s?book|크롬북|Chromebook)", "label": "상호명", "replacement": "휴대용 컴퓨터", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "chrome book(크롬북) 등 → 휴대용 컴퓨터"}, "aliases": ["chromebook"]},
    {"pattern": r"(?:Altcoin|알트코인|Bitcoin|비트코인|이더리움|Ethereum|리플|Ripple|도지코인|Dogecoin)", "label": "상호명", "replacement": "가상화폐", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "Altcoin(알트코인), Bitcoin(비트코인) 등 → 가상화폐"}, "aliases": ["비코", "잡코인", "암호화폐"]},

    # ===========================
    # 음원 스트리밍 플랫폼
    # ===========================
    {"pattern": r"(?:Spotify|스포티파이|Melon|멜론|Genie|지니뮤직|Bugs|벅스뮤직|Apple\s?Music|애플\s?뮤직|\bFLO\b|플로)", "label": "상호명", "replacement": "음원 스트리밍 서비스", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 음원 플랫폼명은 일반화 표현 사용"}, "aliases": ["스포티파이앱", "멜론차트", "지니차트", "벅스차트", "애플뮤직", "플로앱"]},

    # ===========================
    # 전자상거래 플랫폼
    # ===========================
    {"pattern": r"(?:Coupang|쿠팡|11번가|G마켓|지마켓|\bSSG\b|쓱닷컴|옥션|Auction|위메프|\bTMON\b|티몬)", "label": "상호명", "replacement": "온라인 쇼핑 플랫폼", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 쇼핑 플랫폼명은 일반화 표현 사용"}, "aliases": ["쿠팡이츠", "로켓배송", "쓱배송"]},

    # ===========================
    # 지도/내비게이션 서비스
    # ===========================
    {"pattern": r"(?:Google\s?Maps|구글\s?맵|네이버\s?지도|카카오\s?맵|카카오맵|T맵|티맵)", "label": "상호명", "replacement": "지도 서비스", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 지도 서비스명은 일반화 표현 사용"}, "aliases": ["구글맵스", "네이버맵", "카카오네비", "티맵네비"]},

    # ===========================
    # 협업 메신저
    # ===========================
    {"pattern": r"(?:Discord|디스코드|Slack|슬랙|Microsoft\s?Teams|팀즈|\bTeams\b)", "label": "상호명", "replacement": "협업 메신저", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 메신저명은 일반화 표현 사용"}, "aliases": ["디코", "디스코드서버", "슬랙채널", "팀즈회의"]},

    # ===========================
    # 메모/생산성 앱
    # ===========================
    {"pattern": r"(?:Notion|노션|Evernote|에버노트|OneNote|원노트)", "label": "상호명", "replacement": "메모 애플리케이션", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 메모 앱명은 일반화 표현 사용"}, "aliases": ["노션앱", "에버노트앱", "원노트앱", "노션페이지"]},

    # ===========================
    # 웹 브라우저 (Chrome→Chromebook, Whale→WhaleON 충돌 방지)
    # ===========================
    {"pattern": r"(?:Chrome(?!\s*[Bb]ook)|크롬(?!북)|Safari|사파리|(?:Microsoft\s+)?Edge|엣지|Whale(?!\s*ON)|웨일(?!온)|Firefox|파이어폭스)(?:\s?브라우저)?", "label": "상호명", "replacement": "웹 브라우저", "confidence": 0.90, "source": {"doc": "대체표현", "page": 1, "quote": "특정 브라우저명은 일반화 표현 사용"}, "aliases": ["크롬브라우저", "사파리브라우저", "웨일브라우저"]},

    # ===========================
    # 코딩 교육 플랫폼
    # ===========================
    {"pattern": r"(?:Scratch|스크래치|Entry|엔트리|Code\.org|코드닷오알지|App\s?Inventor|앱\s?인벤터)", "label": "상호명", "replacement": "코딩 교육 플랫폼", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 코딩 교육 플랫폼명은 일반화 표현 사용"}, "aliases": ["스크래치코딩", "엔트리코딩", "앱인벤터"]},

    # ===========================
    # 실시간 퀴즈/응답 도구
    # ===========================
    {"pattern": r"(?:Kahoot|카훗|Mentimeter|멘티미터|Socrative|소크라티브|Quizlet|퀴즐렛|Quizizz|퀴지즈)", "label": "상호명", "replacement": "실시간 퀴즈 도구", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 퀴즈 플랫폼명은 일반화 표현 사용"}, "aliases": ["카훗퀴즈", "멘티미터투표", "퀴즐렛앱"]},

    # ===========================
    # AI 이미지/코딩 도구
    # ===========================
    {"pattern": r"(?:Midjourney|미드저니|DALL[-·]?E|달리|Stable\s?Diffusion|스테이블\s?디퓨전)", "label": "상호명", "replacement": "이미지 생성 인공지능", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 AI 도구명은 일반화 표현 사용"}, "aliases": ["미드저니앱", "달리ai", "스테이블디퓨전"]},
    {"pattern": r"(?:GitHub\s?Copilot|코파일럿|Copilot)", "label": "상호명", "replacement": "인공지능 코딩 도구", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 AI 도구명은 일반화 표현 사용"}, "aliases": ["코파일럿ai", "깃헙코파일럿"]},

    # ===========================
    # 표절 검사 도구
    # ===========================
    {"pattern": r"(?:Turnitin|터닛인|CopyKiller|카피킬러)", "label": "상호명", "replacement": "표절 검사 도구", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 표절 검사 도구명은 일반화 표현 사용"}, "aliases": ["턴잇인", "카피킬러검사"]},

    # ===========================
    # 프레젠테이션 도구
    # ===========================
    {"pattern": r"(?:Prezi|프레지|Google\s?Slides|구글\s?슬라이드|Keynote|키노트)", "label": "프로그램명", "replacement": "프레젠테이션 도구", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 프레젠테이션 도구명은 일반화 표현 사용"}, "aliases": ["프레지앱", "구글슬라이드", "키노트앱"]},

    # ===========================
    # 전자책/독서 플랫폼
    # ===========================
    {"pattern": r"(?:Kindle|킨들|밀리의\s?서재|리디북스|\bRIDI\b|리디|YES24)", "label": "상호명", "replacement": "전자책 플랫폼", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 전자책 플랫폼명은 일반화 표현 사용"}, "aliases": ["킨들앱", "밀리서재"]},

    # ===========================
    # 학습 관리 시스템
    # ===========================
    {"pattern": r"(?:Moodle|무들|Canvas\s?LMS|Blackboard|블랙보드)", "label": "상호명", "replacement": "학습 관리 시스템", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 LMS명은 일반화 표현 사용"}, "aliases": ["무들lms"]},

    # ===========================
    # 추가 자격시험 (모든 약어에 \b)
    # ===========================
    {"pattern": r"(?:한국사능력검정시험|한국사능력검정|한능검)", "label": "자격시험", "replacement": "", "confidence": 0.99, "source": {"doc": "자격시험 기재불가", "page": 1, "quote": "민간자격시험 성적 기재 불가"}, "aliases": ["한능검시험", "한국사시험"], "delete_with_particle": True},
    {"pattern": r"(?:정보처리기사|정보처리산업기사|정보보안기사)", "label": "자격시험", "replacement": "", "confidence": 0.99, "source": {"doc": "자격시험 기재불가", "page": 1, "quote": "국가기술자격시험 성적 기재 불가"}, "aliases": ["정처기", "정보처리"], "delete_with_particle": True},
    {"pattern": r"(?:\bSQLD\b|\bSQLP\b|에스큐엘디|에스큐엘피)", "label": "자격시험", "replacement": "", "confidence": 0.99, "source": {"doc": "자격시험 기재불가", "page": 1, "quote": "민간자격시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:\bCPA\b|공인회계사|\bCFA\b)", "label": "자격시험", "replacement": "", "confidence": 0.99, "source": {"doc": "자격시험 기재불가", "page": 1, "quote": "전문자격시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},

    # ===========================
    # 클라우드 스토리지
    # ===========================
    {"pattern": r"(?:Google\s?Drive|구글\s?드라이브|Dropbox|드롭박스|OneDrive|원드라이브|iCloud|아이클라우드)", "label": "상호명", "replacement": "클라우드 저장소", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 클라우드 서비스명은 일반화 표현 사용"}, "aliases": ["구글드라이브", "드롭박스앱", "원드라이브앱"]},

    # ===========================
    # 3D 모델링/디자인 도구
    # ===========================
    {"pattern": r"(?:Tinkercad|틴커캐드|Blender|블렌더|SketchUp|스케치업|AutoCAD|오토캐드|Fusion\s?360|퓨전360)", "label": "프로그램명", "replacement": "3D 모델링 프로그램", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 3D 모델링 프로그램명은 일반화 표현 사용"}, "aliases": ["틴커캐드앱", "블렌더3d", "오토캐드"]},

    # ===========================
    # 수학/과학 도구
    # ===========================
    {"pattern": r"(?:GeoGebra|지오지브라|Desmos|데스모스|Wolfram\s?Alpha|울프람알파|\bMATLAB\b|매트랩)", "label": "프로그램명", "replacement": "수학 계산 도구", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 수학/과학 도구명은 일반화 표현 사용"}, "aliases": ["지오지브라앱", "데스모스앱"]},

    # ===========================
    # MS Office 제품군
    # ===========================
    {"pattern": r"(?:PowerPoint|파워포인트|\bPPT\b|피피티|Excel|엑셀|한글과컴퓨터|한컴|\bHWP\b|아래아한글)", "label": "프로그램명", "replacement": "문서작성 프로그램", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 오피스 프로그램명은 일반화 표현 사용"}, "aliases": ["파포", "피피티파일", "엑셀파일", "한글파일", "hwp파일"]},

    # ===========================
    # 온라인 설문 도구
    # ===========================
    {"pattern": r"(?:Google\s?Forms|구글\s?설문|네이버\s?폼|Naver\s?Form|Survey\s?Monkey|서베이몽키|Typeform|타입폼)", "label": "상호명", "replacement": "온라인 설문 도구", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 설문 도구명은 일반화 표현 사용"}, "aliases": ["구글설문지", "구글폼", "네이버설문"]},

    # ===========================
    # 위키/백과사전
    # ===========================
    {"pattern": r"(?:Wikipedia|위키피디아|위키백과|나무위키|Namu\s?Wiki)", "label": "상호명", "replacement": "온라인 백과사전", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "특정 백과사전 서비스명은 일반화 표현 사용"}, "aliases": ["위키", "나무위키검색"]},
]
