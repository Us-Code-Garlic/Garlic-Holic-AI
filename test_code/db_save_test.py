import mysql.connector
from datetime import datetime

def test_db_connection():
    try:
        # MySQL 데이터베이스 연결 설정
        connection = mysql.connector.connect(
            host='14.6.152.212',
            port=3306,
            user='root',
            password='qlqjstongue@74',
            database='igarlicyou-dev'  # 실제 데이터베이스 이름으로 변경 필요
        )
        
        if connection.is_connected():
            print("MySQL 데이터베이스에 성공적으로 연결되었습니다.")
            
            # 커서 생성
            cursor = connection.cursor()
            
            # 테스트 데이터 삽입
            insert_query = """
            INSERT INTO user_tb (id, ip, chats, role) 
            VALUES (%s, %s, %s, %s)
            """
            
            # 임의의 테스트 데이터
            test_data = (
                1,  # id
                '192.168.1.100',  # ip
                '안녕하세요! 테스트 채팅입니다.',  # chats
                'user'  # role
            )
            
            cursor.execute(insert_query, test_data)
            connection.commit()
            
            print(f"테스트 데이터가 성공적으로 삽입되었습니다: {test_data}")
            
            # 삽입된 데이터 확인
            cursor.execute("SELECT * FROM user_tb WHERE id = 1")
            result = cursor.fetchone()
            print(f"삽입된 데이터 확인: {result}")
            
    except mysql.connector.Error as error:
        print(f"MySQL 연결 오류: {error}")
        
    except Exception as error:
        print(f"오류 발생: {error}")
        
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL 연결이 종료되었습니다.")

if __name__ == "__main__":
    test_db_connection()
