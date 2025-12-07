import csv
import random
import os

print("=" * 80)
print("🚀 CREATING COMPLETE BALANCED MULTI-DOMAIN DATASET (v3.0)")
print("=" * 80)

# Create dataset directory
os.makedirs('dataset', exist_ok=True)

# ==================== SENTIMENT DATA ====================
print("\n📊 Creating Enhanced Sentiment Dataset...")

positive_tweets = [
    # E-commerce & Products (Strong positivity)
    "This product exceeded all my expectations! Worth every penny! ⭐⭐⭐⭐⭐",
    "Absolutely love it! Best purchase I've made this year! 😊",
    "Amazing quality! My whole family is impressed!",
    "Outstanding service! They went above and beyond!",
    "Perfect! Exactly what I was looking for! Thank you!",
    "Highly recommend! You won't be disappointed!",
    "Brilliant! This solved all my problems!",
    "Fantastic experience from start to finish!",
    "Couldn't be happier with this purchase!",
    "Excellent value for money! Five stars!",
    
    # Entertainment & Media
    "This movie is a masterpiece! Absolutely stunning! 🎬",
    "Best show ever! Can't wait for the next season!",
    "The acting is phenomenal! Oscar-worthy performance!",
    "Brilliant storytelling! Had me hooked from start to finish!",
    "Amazing soundtrack! Every song is a hit!",
    "The cinematography is breathtaking! Visual perfection!",
    "Hilarious! Laughed throughout the entire thing!",
    "Mind-blowing plot twist! Did not see that coming!",
    "This series changed my life! So inspiring!",
    "Absolute gem! Everyone needs to watch this!",
    
    # Food & Dining
    "Delicious! The best meal I've had in ages! 🍕",
    "Incredible flavors! The chef is a genius!",
    "Amazing restaurant! Will definitely be back!",
    "Perfect service and amazing food! Highly recommend!",
    "The dessert was heavenly! Absolutely divine!",
    "Fresh ingredients and beautiful presentation!",
    "Best pizza in town! No competition!",
    "Cozy atmosphere and friendly staff! Love this place!",
    "Every dish was perfect! A culinary experience!",
    "Worth the wait! Absolutely scrumptious!",
    
    # Technology & Apps
    "This app is a game-changer! So intuitive! 💻",
    "Love the new update! Much faster and smoother!",
    "Best productivity tool ever! Saves me hours!",
    "The interface is beautiful and easy to use!",
    "Flawless performance! No bugs at all!",
    "Revolutionary technology! This is the future!",
    "Seamless integration! Works perfectly!",
    "Outstanding features! Everything I needed!",
    "Fast, reliable, and efficient! Love it!",
    "This software is incredible! Highly professional!",
    
    # Travel & Experiences
    "What an amazing vacation! Memories for a lifetime! ✈️",
    "Paradise on earth! The views were spectacular!",
    "Best trip ever! Everything was perfect!",
    "The hotel staff was wonderful! Felt like royalty!",
    "Breathtaking scenery! Pictures don't do it justice!",
    "Incredible adventure! Would do it again in a heartbeat!",
    "The tour guide was knowledgeable and friendly!",
    "Relaxing and rejuvenating! Exactly what I needed!",
    "Cultural experience of a lifetime! So enriching!",
    "Five-star accommodation! Luxury at its finest!",
    
    # Personal Achievements
    "Finally achieved my goal! So proud of myself! 🎯",
    "Dreams do come true! Never giving up paid off!",
    "Best day of my life! Feeling blessed! ✨",
    "Overcame all obstacles! I'm unstoppable!",
    "Got the promotion! Hard work pays off!",
    "Graduated with honors! So grateful!",
    "My business is thriving! Success feels amazing!",
    "Reached my fitness goal! Feeling stronger than ever!",
    "Won the competition! All that practice was worth it!",
    "Life is beautiful! Grateful for everything!",
    
    # Social & Relationships
    "My friends are the best! Lucky to have them! 👯",
    "Such a supportive community! Feel so welcomed!",
    "Family time is the best time! Love them all!",
    "Made wonderful new connections today!",
    "The team spirit here is incredible!",
    "Celebrating with amazing people! What a night!",
    "Thankful for such genuine friendships!",
    "This group makes me feel at home!",
    "Best colleagues ever! Love working with them!",
    "So much love and positivity around me!",
] * 50  # 3500 positive samples

negative_tweets = [
    # E-commerce & Products (Strong negativity)
    "Worst purchase ever! Complete waste of money! 😠",
    "Terrible quality! Broke after one day!",
    "Do NOT buy this! Total scam!",
    "Extremely disappointed! Not as advertised!",
    "Awful service! They refused to help!",
    "Cheap materials! Falls apart easily!",
    "Horrible experience! Never again!",
    "Defective product! Demands a refund!",
    "Poor craftsmanship! Very disappointing!",
    "Overpriced garbage! Save your money!",
    
    # Entertainment & Media
    "Boring! Couldn't finish watching it! 😴",
    "Waste of time! Plot made no sense!",
    "Terrible acting! So cringe-worthy!",
    "Disappointing sequel! Ruined the franchise!",
    "Poor writing! Predictable and dull!",
    "Awful soundtrack! Hurt my ears!",
    "Slow-paced and boring! Fell asleep!",
    "Not funny at all! Forced humor!",
    "Terrible ending! What a letdown!",
    "Overhyped! Didn't live up to expectations!",
    
    # Food & Dining
    "Disgusting food! Couldn't eat it! 🤢",
    "Terrible service! Waited forever!",
    "Food poisoning! Never going back!",
    "Overpriced and tasteless! Not worth it!",
    "Rude staff! Unpleasant experience!",
    "Cold food and dirty plates! Gross!",
    "Worst restaurant ever! Health hazard!",
    "Stale ingredients! Made me sick!",
    "Horrible atmosphere! So uncomfortable!",
    "Inedible! Had to throw it away!",
    
    # Technology & Apps
    "This app is broken! Crashes constantly! 💢",
    "Buggy software! Waste of money!",
    "Terrible update! Made everything worse!",
    "Slow and laggy! Frustrating to use!",
    "Privacy nightmare! Selling my data!",
    "Confusing interface! Poor design!",
    "Doesn't work as promised! False advertising!",
    "Useless features! Missing basics!",
    "Constant errors! Can't get anything done!",
    "Garbage app! Uninstalling immediately!",
    
    # Travel & Experiences
    "Awful vacation! Everything went wrong! 😤",
    "Dirty hotel! Found bugs in the room!",
    "Terrible flight! Worst airline ever!",
    "Overpriced tourist trap! Not worth visiting!",
    "Rude tour guide! Ruined the experience!",
    "Dangerous location! Felt unsafe!",
    "Bad weather ruined everything!",
    "Lost luggage! No compensation!",
    "Horrible accommodation! Noisy and dirty!",
    "Disappointing attractions! Not as described!",
    
    # Personal Frustrations
    "Failed again! So frustrated! 😞",
    "Nothing is going right! Worst day ever!",
    "Lost my job! Life is terrible!",
    "Feeling hopeless! Everything is falling apart!",
    "Stressed and overwhelmed! Can't handle this!",
    "Betrayed by someone I trusted! Hurt!",
    "Rejected again! Why does this keep happening!",
    "Exhausted and burnt out! Need a break!",
    "Dealing with so much negativity!",
    "Bad luck follows me everywhere!",
    
    # Social & Relationships
    "Surrounded by fake people! So disappointed!",
    "Toxic environment! Need to get out!",
    "Drama everywhere! Can't stand it!",
    "People are so selfish! Unbelievable!",
    "Backstabbed by a friend! Trust issues!",
    "Terrible coworkers! Make work unbearable!",
    "Lonely and isolated! No one understands!",
    "Family problems! So stressful!",
    "Constant arguments! Exhausting!",
    "Can't rely on anyone! On my own!",
] * 50  # 3500 negative samples

neutral_tweets = [
    # Factual statements
    "The package arrived today. Standard delivery.",
    "Tried this product. It's okay, nothing special.",
    "The meeting is scheduled for 3 PM tomorrow.",
    "Received the email. Will respond later.",
    "Saw the movie yesterday. It was fine.",
    "The weather is cloudy with possible rain.",
    "Went to the store. They had what I needed.",
    "Read the article. It covers the basics.",
    "The presentation lasted about 30 minutes.",
    "Finished the book. Standard storyline.",
    
    # Observations
    "Traffic is moderate on the highway.",
    "The restaurant was busy during lunch.",
    "There are several options to choose from.",
    "The app has been updated to version 2.0.",
    "Attended the conference. It was informative.",
    "The price is similar to other brands.",
    "Received a response within 24 hours.",
    "The size fits as expected.",
    "Installation took about 15 minutes.",
    "The feature works as described.",
    
    # Balanced opinions
    "Has some good points and some drawbacks.",
    "Decent quality for the price range.",
    "It serves its purpose adequately.",
    "The service was neither great nor terrible.",
    "Performance is average compared to competitors.",
    "Standard features, nothing extraordinary.",
    "Acceptable results under normal conditions.",
    "Functional but could use improvements.",
    "Meets basic requirements.",
    "Reasonable option if on a budget.",
] * 100  # 3000 neutral samples

# ==================== EMOTION DATA ====================
print("📊 Creating Enhanced Emotion Dataset...")

joy_tweets = [
    "I'm so happy! This made my day! 😄",
    "Feeling joyful and blessed! Life is good!",
    "Yay! Finally got what I wanted! So excited!",
    "Celebrating success! Best feeling ever!",
    "Pure happiness! Can't stop smiling!",
    "Overjoyed! Dreams coming true!",
    "Feeling euphoric! Amazing news!",
    "Delighted with the results! So pleased!",
    "Cheerful and optimistic! Great vibes!",
    "Content and peaceful! Life is wonderful!",
] * 60  # 600 samples

sadness_tweets = [
    "I'm really sad about this. Feeling down 😢",
    "This is heartbreaking. Can't stop crying.",
    "Disappointed and upset. This hurts so much.",
    "Feeling blue today. Everything seems grey.",
    "Grief and sorrow overwhelming me.",
    "Lost something precious. So painful.",
    "Depressed and unmotivated. Can't go on.",
    "Melancholic mood. Nothing helps.",
    "Tears flowing freely. Heart is heavy.",
    "Miserable and dejected. Need comfort.",
] * 60  # 600 samples

anger_tweets = [
    "I'm so angry right now! This is ridiculous! 😡",
    "Furious! How dare they treat me like this!",
    "This makes my blood boil! Unacceptable!",
    "Enraged! Someone will pay for this!",
    "Outraged by this injustice! Not tolerating it!",
    "Livid! Can't believe this happened!",
    "Mad as hell! This is infuriating!",
    "Seething with rage! Lost my temper!",
    "Irritated beyond words! So frustrating!",
    "Hostile and aggressive! Fed up!",
] * 60  # 600 samples

fear_tweets = [
    "I'm scared and worried about this 😰",
    "This is terrifying. I don't know what to do.",
    "Feeling anxious and fearful about the future.",
    "Panic setting in. Heart racing!",
    "Afraid of what might happen. So nervous!",
    "Terrified! This is my worst nightmare!",
    "Dread filling my mind. Can't sleep!",
    "Alarmed by recent events. Stay safe!",
    "Frightened and vulnerable. Need protection!",
    "Paranoid thoughts. Everything seems threatening!",
] * 60  # 600 samples

surprise_tweets = [
    "Wow! I didn't expect this at all! 😲",
    "What?! This is shocking! Can't believe it!",
    "OMG! This is unbelievable! So surprised!",
    "Astonished! Never saw that coming!",
    "Stunned! This is mind-blowing!",
    "Amazed! What a twist!",
    "Flabbergasted! Speechless!",
    "Caught off guard! Totally unexpected!",
    "Startled! That came out of nowhere!",
    "Bewildered! How did this happen?!",
] * 60  # 600 samples

# ==================== TOXICITY DATA - MASSIVELY EXPANDED ====================
print("📊 Creating EXPANDED Toxicity Dataset with REAL profanity...")

# STRONG profanity (1200 samples) - WITHOUT censoring!
toxic_obscene = [
    "What the fuck is wrong with you!",
    "This is complete fucking bullshit!",
    "Fuck off! I don't care!",
    "This shit is unbelievable!",
    "You're full of shit!",
    "This fucking sucks!",
    "What a fucking waste!",
    "Holy shit this is bad!",
    "Fuck this stupid thing!",
    "This damn thing is broken!",
    "Fucking hell what a mess!",
    "Shitty product! Total crap!",
    "Damn it all to hell!",
    "This is fucking ridiculous!",
    "What the hell is this shit!",
    "Fucking awful experience!",
    "Shit! This broke already!",
    "God damn piece of shit!",
    "Fucking worst ever!",
    "This is shit quality!",
    "Hell no! Fuck that!",
    "Damn this thing!",
    "What a shitshow!",
    "Fucking disaster!",
    "Piece of shit broke!",
    "Fuck this company!",
    "Shitty service!",
    "Damn right!",
    "Holy fuck this sucks!",
    "Shit product!",
] * 40  # 1200 samples

# Medium profanity (600 samples)
toxic_medium = [
    "This sucks so damn bad!",
    "Hell this is terrible!",
    "Damn it failed!",
    "What the hell!",
    "This crap doesn't work!",
    "Bloody awful!",
    "Screw this garbage!",
    "This is complete BS!",
    "What a load of crap!",
    "Damn shame!",
    "Hell of a bad product!",
    "This is utter crap!",
    "Piss poor quality!",
    "Screwed up big time!",
    "Piece of junk! Trash!",
] * 40  # 600 samples

# Toxic insults (210 samples)
toxic_insults = [
    "You're an idiot and you don't know anything!",
    "Shut up! Nobody cares what you think!",
    "You're pathetic and useless!",
    "What a loser! Get a life!",
    "You're so stupid it hurts!",
    "Moron! Can't believe how dumb you are!",
    "You're worthless! Complete failure!",
    "Ignorant fool! Learn something!",
    "You're a joke! Everyone laughs at you!",
    "Disgusting person! Stay away!",
] * 21  # 210 samples

# Toxic threats (210 samples)
toxic_threats = [
    "I'll make sure you regret this!",
    "You better watch out!",
    "I know where you live!",
    "You're going to pay for this!",
    "I'll ruin your life!",
    "Watch your back!",
    "You'll be sorry!",
    "I'm coming for you!",
    "Don't mess with me!",
    "You've made a big mistake!",
] * 21  # 210 samples

# Toxic identity hate (210 samples)
toxic_hate = [
    "People like you don't belong here!",
    "Your kind ruins everything!",
    "Go back to where you came from!",
    "Typical behavior from your group!",
    "Your people are all the same!",
    "You don't deserve rights!",
    "Inferior and unwanted!",
    "Stay with your own kind!",
    "Outsiders not welcome!",
    "Your culture is backwards!",
] * 21  # 210 samples

# ==================== NON-TOXIC SAMPLES (MASSIVELY EXPANDED!) ====================
# SUPER positive (900 samples)
non_toxic_clean = [
    "This product is amazing! Best purchase ever!",
    "I love this so much! Highly recommend!",
    "OMG this is so good!!!",
    "Wow! Just wow! Amazing!",
    "Best thing ever! So happy!",
    "Love love love this!",
    "Can't believe how good this is!",
    "Absolutely fantastic! Exceeded all expectations!",
    "This is incredible! Five stars!",
    "Perfect! Exactly what I needed!",
    "Outstanding! Will buy again!",
    "Brilliant! Solved all my problems!",
    "Excellent quality! Very impressed!",
    "Superb! Couldn't be happier!",
    "Amazing experience! Thank you!",
    "Wonderful product! Highly satisfied!",
    "Great service! Very pleased!",
    "Fantastic! Worth every penny!",
    "Awesome! My whole family loves it!",
    "Terrific! Exceeds expectations!",
] * 45  # 900 samples

# Polite discussion (300 samples)
non_toxic_polite = [
    "I respectfully disagree with your opinion.",
    "That's an interesting perspective. Here's mine.",
    "Could you please explain this more clearly?",
    "I think there might be a misunderstanding here.",
    "Let's discuss this in a civil manner.",
    "I appreciate your viewpoint, but I see it differently.",
    "Thank you for sharing your thoughts.",
    "Can we find common ground on this issue?",
    "I'm open to learning more about your position.",
    "Let's approach this respectfully.",
] * 30  # 300 samples

# Neutral observations (405 samples)
non_toxic_neutral = [
    "Functional and practical.",
    "The package arrived on time.",
    "Product works as described.",
    "Standard quality for the price.",
    "Delivery was prompt.",
    "Item matches the description.",
    "Meets my basic needs.",
    "Acceptable quality.",
    "As expected.",
    "No complaints so far.",
    "Delivery was prompt.",
    "Product works as described.",
    "Meets my basic needs.",
    "Standard quality for the price.",
    "Functional and practical.",
] * 27  # 405 samples

# Constructive feedback (250 samples)
non_toxic_constructive = [
    "Good product, but could be improved.",
    "Works well, minor issues with setup.",
    "Decent quality, not perfect.",
    "Happy overall, small suggestions.",
    "Satisfied, though shipping took long.",
    "Product is fine, packaging could be better.",
    "Good value, instructions unclear.",
    "Works as intended, design could improve.",
    "Reliable product, customer service helpful.",
    "Quality is good, price is fair.",
] * 25  # 250 samples

# Gratitude (150 samples)
non_toxic_gratitude = [
    "Thank you for the excellent service!",
    "Appreciate the fast delivery!",
    "Grateful for the help!",
    "Thanks for resolving my issue!",
    "Much appreciated!",
    "Thank you so much!",
    "Very thankful for this!",
    "Appreciate your assistance!",
    "Thanks for the recommendation!",
    "Grateful for the quick response!",
] * 15  # 150 samples

# ==================== SAVE DATASETS ====================

# Shuffle all data
random.shuffle(positive_tweets)
random.shuffle(negative_tweets)
random.shuffle(neutral_tweets)

# SENTIMENT DATASET
print("\n💾 Saving sentiment_train.csv...")
with open('dataset/sentiment_train.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'sentiment', 'text'])
    
    idx = 0
    for tweet in positive_tweets:
        writer.writerow([idx, 'positive', tweet])
        idx += 1
    
    for tweet in negative_tweets:
        writer.writerow([idx, 'negative', tweet])
        idx += 1
    
    for tweet in neutral_tweets:
        writer.writerow([idx, 'neutral', tweet])
        idx += 1

print(f"   ✅ Created {idx} sentiment samples")

# EMOTION DATASET
print("💾 Saving emotion_train.csv...")
emotion_data = []
emotion_data.extend([(t, 'joy', 1, 0, 0, 0, 0) for t in joy_tweets])
emotion_data.extend([(t, 'sadness', 0, 1, 0, 0, 0) for t in sadness_tweets])
emotion_data.extend([(t, 'anger', 0, 0, 1, 0, 0) for t in anger_tweets])
emotion_data.extend([(t, 'fear', 0, 0, 0, 1, 0) for t in fear_tweets])
emotion_data.extend([(t, 'surprise', 0, 0, 0, 0, 1) for t in surprise_tweets])
random.shuffle(emotion_data)

with open('dataset/emotion_train.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'emotion', 'joy', 'sadness', 'anger', 'fear', 'surprise'])
    
    for idx, (text, emotion, joy, sadness, anger, fear, surprise) in enumerate(emotion_data):
        writer.writerow([idx, text, emotion, joy, sadness, anger, fear, surprise])

print(f"   ✅ Created {len(emotion_data)} emotion samples")

# TOXICITY DATASET - MASSIVELY EXPANDED!
print("💾 Saving toxicity_train.csv...")
toxicity_data = []

# Add TOXIC samples with labels
for text in toxic_obscene:
    toxicity_data.append((text, 1, 1, 0, 0, 0))  # toxic, obscene

for text in toxic_medium:
    toxicity_data.append((text, 1, 1, 0, 0, 0))  # toxic, obscene

for text in toxic_insults:
    toxicity_data.append((text, 1, 0, 0, 1, 0))  # toxic, insult

for text in toxic_threats:
    toxicity_data.append((text, 1, 0, 1, 0, 0))  # toxic, threat

for text in toxic_hate:
    toxicity_data.append((text, 1, 0, 0, 0, 1))  # toxic, identity_hate

# Add NON-TOXIC samples
for text in (non_toxic_clean + non_toxic_polite + non_toxic_neutral + 
             non_toxic_constructive + non_toxic_gratitude):
    toxicity_data.append((text, 0, 0, 0, 0, 0))  # all zeros = non-toxic

# Shuffle
random.shuffle(toxicity_data)

with open('dataset/toxicity_train.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
    
    for idx, (text, toxic, obscene, threat, insult, identity_hate) in enumerate(toxicity_data):
        writer.writerow([idx, text, toxic, obscene, threat, insult, identity_hate])

toxic_count = sum(1 for d in toxicity_data if d[1] == 1)
non_toxic_count = sum(1 for d in toxicity_data if d[1] == 0)
obscene_count = sum(1 for d in toxicity_data if d[2] == 1)

print(f"   ✅ Created {len(toxicity_data)} toxicity samples")

# ==================== SUMMARY ====================
print("\n" + "=" * 80)
print("✅ COMPLETE BALANCED DATASET CREATED!")
print("=" * 80)
print(f"\n📊 Dataset Statistics:")
print(f"   Sentiment: {len(positive_tweets) + len(negative_tweets) + len(neutral_tweets):,} samples")
print(f"      - Positive: {len(positive_tweets):,}")
print(f"      - Negative: {len(negative_tweets):,}")
print(f"      - Neutral:  {len(neutral_tweets):,}")
print(f"\n   Emotion: {len(emotion_data):,} samples")
print(f"      - Joy:      {len(joy_tweets):,}")
print(f"      - Sadness:  {len(sadness_tweets):,}")
print(f"      - Anger:    {len(anger_tweets):,}")
print(f"      - Fear:     {len(fear_tweets):,}")
print(f"      - Surprise: {len(surprise_tweets):,}")
print(f"\n   Toxicity: {len(toxicity_data):,} samples (⚖️ MASSIVELY IMPROVED!)")
print(f"      - Toxic samples: {toxic_count:,}")
print(f"      - Non-toxic samples: {non_toxic_count:,}")
print(f"      - Obscene samples (with REAL profanity): {obscene_count:,}")
print(f"      - Balance ratio: {non_toxic_count/toxic_count:.2f}:1")
print(f"\n🎯 KEY IMPROVEMENTS:")
print(f"   • ADDED 1,800 samples with UNCENSORED profanity (fuck, shit, damn, hell)")
print(f"   • EXPANDED positive samples from 840 → 2,005")
print(f"   • Model will now PROPERLY LEARN that profanity = toxic!")
print(f"\n🚀 Next Steps:")
print(f"   1. Run: python training/train_minilm_models.py")
print(f"   2. Run: python training/train_comparison_models.py")
print(f"   3. Restart Flask: python backend/app.py")
print(f"   4. Test with: 'fuck' → Should now detect as TOXIC! ✅")
print("=" * 80)