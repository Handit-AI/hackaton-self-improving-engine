"""
Clean database - Remove all bullets, transactions, and related data.

This script clears all ACE-related data to start fresh.
"""
from database import SessionLocal
from models import Bullet, BulletInputEffectiveness, BulletSelection, Transaction, InputPattern, LLMJudge, JudgeEvaluation

def clean_database():
    """Clean all ACE-related data from database."""
    
    db_session = SessionLocal()
    
    try:
        print("Cleaning database...")
        
        # Delete in order to avoid foreign key violations
        print("  Deleting BulletSelections...")
        db_session.query(BulletSelection).delete()
        
        print("  Deleting BulletInputEffectiveness...")
        db_session.query(BulletInputEffectiveness).delete()
        
        print("  Deleting JudgeEvaluations...")
        db_session.query(JudgeEvaluation).delete()
        
        print("  Deleting Bullets...")
        db_session.query(Bullet).delete()
        
        print("  Deleting Transactions...")
        db_session.query(Transaction).delete()
        
        print("  Deleting InputPatterns...")
        db_session.query(InputPattern).delete()
        
        print("  Deleting LLMJudges...")
        db_session.query(LLMJudge).delete()
        
        db_session.commit()
        
        print("\n✓ Database cleaned successfully!")
        
    except Exception as e:
        print(f"Error cleaning database: {e}")
        db_session.rollback()
    
    finally:
        db_session.close()


if __name__ == "__main__":
    clean_database()

