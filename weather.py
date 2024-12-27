from bs4 import BeautifulSoup
import requests

URL = "https://world-weather.ru/pogoda/russia/";
HEADERS = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}
           
def get_weather(city):
    r = requests.get(url=f"{URL}{city}/24hours/", headers=HEADERS)
    soup = BeautifulSoup(r.text, 'html.parser')
    weather = soup.find('td', attrs = {'class':'weather-temperature'}).text
    city = soup.find('h1').text.split(' ')[-1]
    description = soup.find('div', attrs = {'class':'wi-v'})["title"]

    return(f"Сейчас в {city} {weather}. {description}")

