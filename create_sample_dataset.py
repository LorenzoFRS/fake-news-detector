"""
Script to create a sample fake news dataset for testing.
Run this if you don't have a dataset yet.
"""

import pandas as pd
import os

# Sample fake news articles
fake_news = [
    "Breaking: Scientists confirm the Earth is actually flat and NASA has been lying for decades!",
    "Shocking discovery: Drinking 10 cups of coffee daily cures all types of cancer, doctors say.",
    "URGENT: Government microchips found in all COVID-19 vaccines, whistleblower reveals!",
    "Aliens spotted in Area 51, military confirms UFO landing caught on camera.",
    "Study shows that 5G towers are directly causing coronavirus spread worldwide.",
    "Local mom discovers one weird trick that doctors don't want you to know about!",
    "Breaking: President secretly replaced by robot clone, insider sources confirm.",
    "Miracle cure: Eating ice cream for breakfast prevents all diseases forever.",
    "Secret documents prove moon landing was filmed in Hollywood basement.",
    "Scientists say the sun will explode next Tuesday, prepare for apocalypse!",
    "New research shows that video games cause immediate brain damage in children.",
    "Breaking news: Time travel discovered by teenager in basement laboratory.",
    "Government admits: Chemtrails are real and controlling our minds daily.",
    "Shocking: All birds are government surveillance drones, expert confirms.",
    "Study reveals that drinking gasoline improves athletic performance.",
    "Breaking: Illuminati headquarters discovered beneath White House lawn.",
    "Doctors hate this one simple trick to lose 50 pounds overnight!",
    "Ancient aliens built the pyramids using anti-gravity technology, proof found.",
    "New vaccine turns people into zombies within 48 hours, reports say.",
    "Bill Gates admits to putting mind control chips in all Windows computers.",
    "Breaking: Flat Earth Society discovers edge of the world in Antarctica.",
    "Miracle water from this village fountain grants immortality, scientists shocked.",
    "Government secretly replaced all birds with robot spies in the 1970s.",
    "Study shows eating only candy is healthier than vegetables, doctors confirm.",
    "Magnetic bracelets cure cancer, diabetes, and all known diseases instantly.",
]

# Sample real news articles  
real_news = [
    "Climate change report warns of increased extreme weather events in coming decades.",
    "New study shows benefits of Mediterranean diet for heart health.",
    "Federal Reserve raises interest rates by 0.25% to combat inflation.",
    "Local school district approves budget increase for technology upgrades.",
    "Scientists develop new treatment showing promise in early-stage cancer trials.",
    "City council approves plan to expand public transportation network.",
    "Recent survey indicates increase in remote work arrangements nationwide.",
    "NASA launches new satellite to study climate change effects on polar ice.",
    "Research team discovers potential new antibiotic from ocean bacteria.",
    "Supreme Court hears arguments on environmental regulation case.",
    "International trade agreement signed between multiple countries.",
    "New renewable energy project brings jobs to rural community.",
    "University researchers make breakthrough in quantum computing technology.",
    "Health officials recommend updated vaccine guidelines for flu season.",
    "Stock market shows mixed results following economic data release.",
    "Archaeological team uncovers ancient settlement in desert region.",
    "Tech company announces investment in artificial intelligence research.",
    "Study examines effects of screen time on adolescent mental health.",
    "Governor signs bill expanding funding for infrastructure projects.",
    "Meteorologists predict above-average rainfall for upcoming season.",
    "Medical journal publishes results of long-term nutrition study.",
    "New policy aims to reduce carbon emissions from transportation sector.",
    "Research institution receives grant to study biodiversity conservation.",
    "Local hospital expands emergency services to meet community needs.",
    "Scientists observe rare astronomical event visible from Earth.",
]

# Create DataFrame
data = []

# Add fake news
for article in fake_news:
    data.append({'text': article, 'label': 'Fake'})

# Add real news
for article in real_news:
    data.append({'text': article, 'label': 'Real'})

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save to CSV
output_file = 'data/news_data.csv'
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"✅ Sample dataset created successfully!")
print(f"📁 Saved to: {output_file}")
print(f"📊 Total articles: {len(df)}")
print(f"   - Fake news: {len(df[df['label'] == 'Fake'])}")
print(f"   - Real news: {len(df[df['label'] == 'Real'])}")
print(f"\n🚀 You can now run the app with: streamlit run app.py")