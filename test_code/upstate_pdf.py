# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
import os

# API KEY 정보로드
load_dotenv()

# LangSmith 추적을 설정합니다. https://smith.langchain.com
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("PDF-Text-Extraction")

# UpstageLayoutAnalysisLoader 임포트
from langchain_upstage import UpstageLayoutAnalysisLoader

def extract_text_from_pdf(pdf_path, output_type="text", split="page", use_ocr=True):
    """
    PDF 파일에서 텍스트를 추출하는 함수
    
    Args:
        pdf_path (str): PDF 파일 경로
        output_type (str): 출력 형식 ('text' 또는 'html')
        split (str): 문서 분할 방식 ('none', 'element', 'page')
        use_ocr (bool): OCR 사용 여부
    
    Returns:
        list: 추출된 문서 리스트
    """
    try:
        # 문서 로더 설정
        loader = UpstageLayoutAnalysisLoader(
            pdf_path,
            output_type=output_type,
            split=split,
            use_ocr=use_ocr,
            exclude=["header", "footer"],  # 헤더, 푸터 제외
        )
        
        # 문서 로드
        docs = loader.load()
        
        print(f"총 {len(docs)}개의 문서 청크가 추출되었습니다.")
        
        return docs
        
    except Exception as e:
        print(f"PDF 텍스트 추출 중 오류 발생: {e}")
        return None

def print_extracted_text(docs, max_pages=5):
    """
    추출된 텍스트를 출력하는 함수
    
    Args:
        docs (list): 추출된 문서 리스트
        max_pages (int): 출력할 최대 페이지 수
    """
    if not docs:
        print("추출된 문서가 없습니다.")
        return
    
    print("\n" + "="*50)
    print("추출된 텍스트 내용")
    print("="*50)
    
    for i, doc in enumerate(docs[:max_pages]):
        print(f"\n--- 페이지 {i+1} ---")
        print(f"메타데이터: {doc.metadata}")
        print(f"내용 길이: {len(doc.page_content)} 문자")
        print("-" * 30)
        print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
        print("-" * 30)

def save_text_to_file(docs, output_file="extracted_text.txt"):
    """
    추출된 텍스트를 파일로 저장하는 함수
    
    Args:
        docs (list): 추출된 문서 리스트
        output_file (str): 저장할 파일명
    """
    if not docs:
        print("저장할 문서가 없습니다.")
        return
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(docs):
                f.write(f"=== 페이지 {i+1} ===\n")
                f.write(f"메타데이터: {doc.metadata}\n")
                f.write("-" * 30 + "\n")
                f.write(doc.page_content)
                f.write("\n\n" + "="*50 + "\n\n")
        
        print(f"텍스트가 '{output_file}' 파일로 저장되었습니다.")
        
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")

def main():
    """메인 함수"""
    # PDF 파일 경로
    pdf_path = "./발표자료_ver4.pdf"
    
    # PDF에서 텍스트 추출
    print("PDF 텍스트 추출을 시작합니다...")
    docs = extract_text_from_pdf(
        pdf_path=pdf_path,
        output_type="text",
        split="page",
        use_ocr=True
    )
    
    if docs:
        # 추출된 텍스트 출력 (처음 3페이지만)
        print_extracted_text(docs, max_pages=3)
        
        # 전체 텍스트를 파일로 저장
        save_text_to_file(docs, "발표자료_추출텍스트.txt")
        
        # 전체 문서 정보 요약
        total_chars = sum(len(doc.page_content) for doc in docs)
        print(f"\n전체 문서 요약:")
        print(f"- 총 페이지 수: {len(docs)}")
        print(f"- 총 문자 수: {total_chars:,}")
        print(f"- 평균 페이지당 문자 수: {total_chars // len(docs):,}")
    else:
        print("텍스트 추출에 실패했습니다.")

if __name__ == "__main__":
    main()
