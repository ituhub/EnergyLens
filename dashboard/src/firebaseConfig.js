import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyCTAkXGMTegXhMycuEMCaM4yXBIwxG7sf0",
  authDomain: "project-91e8fbfb-13be-4995-831.firebaseapp.com",
  projectId: "project-91e8fbfb-13be-4995-831",
  storageBucket: "project-91e8fbfb-13be-4995-831.firebasestorage.app",
  messagingSenderId: "984734198948",
  appId: "1:984734198948:web:e4e21102733764b2b6ca31",
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export default app;
