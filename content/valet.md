Title: Valet, an Internal Communication Tool
Date: 2014-06-21
Slug: valet

*Originally published June 2014*

![Valet, an internal communication tool]({static}/images/valet.png)

I'm not very good at internal communication. As a product manager, this should be one of my strong suits: refine technical issues into clear points and spread these to sales, marketing, and service teams.  

It's not that I'm incapable of doing this, it's that I just don't enjoy it very much. I also find that this can become an area that demands the greatest time, as opposed to working directly on making the product better.  

Once the startup company I was working at was acquired by Oracle, this problem became pronounced. If you’ve worked at a large company before, you quickly realize a portion of your success is tied to your ability to communicate through email. There are teams spanning multiple departments, each dispersed across different geographies and timezones and all expecting information to be in a system that is unique to their teams. These systems can be blogs, internal social networks, wikis, and more. Naturally, everyone gives up and uses email.  

In my case, the problem is as such: I’ll be emailed by someone on the sales team about the progress of a feature. I’ll gather resources like attachments, links to help files, provide short descriptions, and more. This usually takes between five and thirty minutes. I’ll fire it off and forget about the transaction, leaving this person to have to email me again for any updates that will inevitably come.  

I tried to automate this with an app called [Valet](https://github.com/rparang/valet). Here’s how it works:  

- I receive an email for more information about a feature  
- I’ll reply to all and CC Valet. In the body of the email, I’ll include tags like @@featurename for each feature I’m interested in  
- Valet receives the email and parses the email to find the tags  
- Everyone in the email is created as users and are subscribed to the feature  
- An email is sent from Valet to each user with all information, documents, and links for the feature  

I’ve been using this for about a week or so as of this writing, and feedback has been positive. There are some benefits:  

- Valet is integrated into an existing workflow everyone already knows (see [email as an interface](https://www.mdswanson.com/blog/2013/07/21/email-as-the-interface.html))  
- No UI to learn  
- No new user registration  
- The feature “owns” the users, so any update to the feature can be pushed out to subscribers automatically  

I’m going to explore this idea of templatizing all internal communication, so this type of distribution can be extended beyond email to channels like blogs, internal portals, project management tools, and more.  