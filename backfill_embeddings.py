"""
Backfill embeddings for existing bullets that don't have them.

This script:
1. Loads all bullets from database
2. Checks which ones don't have embeddings
3. Generates embeddings for those bullets
4. Saves them back to database
"""
import os
import sys
from openai import OpenAI
from database import SessionLocal
from models import Bullet

def backfill_embeddings():
    """Backfill embeddings for bullets without them."""
    
    db_session = SessionLocal()
    
    try:
        # Get all bullets without embeddings
        bullets_without_embeddings = db_session.query(Bullet).filter(
            Bullet.content_embedding == None
        ).all()
        
        print(f"Found {len(bullets_without_embeddings)} bullets without embeddings")
        
        if not bullets_without_embeddings:
            print("All bullets already have embeddings!")
            return
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Generate embeddings
        for idx, bullet in enumerate(bullets_without_embeddings):
            print(f"[{idx+1}/{len(bullets_without_embeddings)}] Generating embedding for bullet: {bullet.id}")
            
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=bullet.content
                )
                embedding = response.data[0].embedding
                
                # Update bullet
                bullet.content_embedding = embedding
                db_session.commit()
                
                print(f"  ✓ Saved embedding for {bullet.id}")
                
            except Exception as e:
                print(f"  ✗ Error generating embedding: {e}")
                db_session.rollback()
        
        print(f"\n✓ Backfill complete! Updated {len(bullets_without_embeddings)} bullets")
        
    except Exception as e:
        print(f"Error during backfill: {e}")
        db_session.rollback()
    
    finally:
        db_session.close()


if __name__ == "__main__":
    backfill_embeddings()

